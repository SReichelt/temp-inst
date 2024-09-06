#![cfg_attr(not(feature = "std"), no_std)]

use core::{
    cmp::Ordering,
    fmt::{self, Debug},
    future::Future,
    hash::{Hash, Hasher},
    marker::{PhantomData, PhantomPinned},
    ops::{Deref, DerefMut, Range, RangeFrom, RangeFull, RangeTo},
    pin::{pin, Pin},
    ptr::NonNull,
    slice,
    str::Chars,
};

use {mapped::*, wrappers::*};

/// A safe lifetime-erased wrapper for an object with lifetime parameters. Intended for use with
/// APIs that require a single (shared or mutable) reference to an object that cannot depend on
/// lifetimes (except for the lifetime of that single reference).
///
/// For example, [`TempInst`] can be used to contract a tuple of references into a single reference
/// and then later back into a tuple of references. For the full list of types that can be used
/// with [`TempInst`], see the implementations of the [`TempRepr`] trait.
///
/// # Examples
///
/// ```
/// # use crate::temp_inst::*;
///
/// // We want to use this external function, which takes a single mutable reference to some `T`
/// // (with a useless `'static` bound because otherwise there will be a trivial solution).
/// fn run_twice<T: 'static>(obj: &mut T, f: fn(&mut T)) {
///     f(obj);
///     f(obj);
/// }
///
/// // However, here we have two separate variables, and we want to pass references to both of them
/// // to `run_twice`. `T = (&mut a, &mut b)` won't work because of the `'static` bound. (In a
/// // real-world use case, the problem is usually that `T` cannot have lifetime parameters for some
/// // other reason.)
/// let mut a = 42;
/// let mut b = 23;
///
/// // The lambda we pass to `call_with_mut` receives a single mutable reference
/// // `inst: &mut TempInst<_>` that we can use.
/// TempInst::<(TempRefMut<i32>, TempRefMut<i32>)>::call_with_mut((&mut a, &mut b), |inst| {
///     run_twice(inst, |inst| {
///         // Now that we have passed `inst` through the external `run_twice` function back to our
///         // own code, we can extract the original pair of references from it.
///         let (a_ref, b_ref) = inst.get_mut();
///         *a_ref += *b_ref;
///         *b_ref += 1;
///     })
/// });
///
/// assert_eq!(a, 42 + 23 + 1 + 23);
/// assert_eq!(b, 23 + 1 + 1);
/// ```
///
/// For shared or pinned mutable references, there is a slightly simpler API:
///
/// ```
/// # use core::ops::Add;
/// # use crate::temp_inst::*;
///
/// // Again, an external function that we want to call with a reference.
/// fn run_twice_and_add<T: 'static>(obj: &T, f: fn(&T) -> i32) -> i32 {
///     f(obj) + f(obj)
/// }
///
/// let a = 42;
/// let b = 23;
///
/// // We want to use `T = (&a, &b)`, but can't because of the `'static` bound.
/// // Instead, we can create a `TempInst` wrapper (that implements `Deref`).
/// let inst = TempInst::<(TempRef<i32>, TempRef<i32>)>::new_wrapper((&a, &b));
/// let sum_of_products = run_twice_and_add(&*inst, |inst| {
///     // Now that have we passed `inst` through the external `run_twice_and_add` function back to
///     // our own own code, we can extract the original pair of references from it.
///     let (a_ref, b_ref) = inst.get();
///     *a_ref * *b_ref
/// });
///
/// assert_eq!(sum_of_products, 42 * 23 + 42 * 23);
/// ```
pub struct TempInst<T: TempRepr>(T, PhantomPinned);

impl<T: TempRepr> TempInst<T> {
    /// Converts an instance of `T::Shared` into a wrapper that implements
    /// [`Deref<Target = TempInst>`]. Note that `T::Shared` is always the non-mutable variant of
    /// `T`; e.g. even if `T` is [`TempRefMut<X>`], `T::Shared` is `&X`, not `&mut X`.
    ///
    /// Afterwards, an instance of `T::Shared` can be recovered from the [`TempInst`] reference
    /// via [`Self::get`].
    pub fn new_wrapper(obj: T::Shared<'_>) -> TempInstWrapper<T> {
        TempInstWrapper::new(obj)
    }

    /// Calls `f` with a shared reference to a [`TempInst`] instance constructed from `obj`, and
    /// returns the value returned by `f`.
    ///
    /// This method exists for consistency (with respect to [`Self::call_with_mut`]), but is
    /// actually just a trivial application of [`Self::new_wrapper`].
    pub fn call_with<R>(obj: T::Shared<'_>, f: impl FnOnce(&Self) -> R) -> R {
        let inst = Self::new_wrapper(obj);
        f(&inst)
    }

    /// Returns the object that was originally passed to [`Self::new`], [`Self::new_wrapper`], etc.,
    /// with a lifetime that is restricted to that of `self`.
    pub fn get(&self) -> T::Shared<'_> {
        // SAFETY:
        //
        // Instances of `TempInst<T>` can only be created via `new` or `new_wrapper[...]`, and all
        // do so via `T::shared_to_temp` or `T::mut_to_temp`.
        //
        // `new` requires a static lifetime. `new_wrapper[...]` accepts any lifetime, but the
        // returned wrapper has the same lifetime, effectively borrowing all contained references
        // until the wrapper is dropped. Dereferencing the wrapper borrows it, so we can be sure
        // that the lifetime of the `TempInst` reference is outlived by the lifetime originally
        // passed to `shared_to_temp` (or `mut_to_temp`).
        unsafe { self.0.temp_to_shared() }
    }
}

impl<T: TempReprMut> TempInst<T> {
    /// Creates a new [`TempInst`] instance from an object with a static lifetime. This is the only
    /// way to create an owned instance instead of a temporary wrapper.
    ///
    /// For safety reasons, [`Self::new`] receives an instance of `T::Mutable` rather than
    /// `T::Shared`, but note that if `T` is e.g. [`TempRef`] rather than [`TempRefMut`],
    /// `T::Mutable` is actually the same as `T::Shared`.
    pub fn new(obj: T::Mutable<'static>) -> Self {
        TempInst(T::mut_to_temp(obj), PhantomPinned)
    }

    /// Converts an instance of `T::Mutable` into a wrapper that implements
    /// [`DerefMut<Target = TempInst>`].
    ///
    /// Afterwards, an instance of `T::Mutable` can be recovered from the [`TempInst`] reference
    /// via [`Self::get_mut`].
    ///
    /// As this method is unsafe, using one of the safe alternatives is strongly recommended:
    /// * [`Self::new_wrapper_pin`] if a pinned mutable [`TempInst`] reference is sufficient.
    /// * [`Self::call_with_mut`] if the use of the mutable [`TempInst`] reference can be confined
    ///   to a closure.
    ///
    /// [`Self::new_wrapper_mut`] potentially has a slight overhead compared to
    /// [`Self::new_wrapper_pin`], in terms of both time and space, though there is a good chance
    /// that the compiler will optimize both away if it can see how the wrapper is used.
    ///
    /// # Safety
    ///
    /// The caller must ensure at least one of the following two conditions.
    /// * The [`Drop`] implementation of the returned wrapper is called whenever it goes out of
    ///   scope. (In particular, the wrapper must not be passed to [`core::mem::forget`].)
    /// * The state of the wrapper when it goes out of scope is the same as when it was created.
    ///   (This condition can be violated by calling [`core::mem::swap`] or a related function on
    ///   the [`TempInst`] reference. When the wrapper goes out of scope after passing the
    ///   [`TempInst`] reference to [`core::mem::swap`], the _other_ [`TempInst`] instance that it
    ///   was swapped with can become dangling. Note that passing the result of [`Self::get_mut`] to
    ///   [`core::mem::swap`] is not unsafe.)
    ///
    /// # Panics
    ///
    /// The [`Drop`] implementation of the returned wrapper calls [`std::process::abort`] (after
    /// calling the standard panic handler) if the [`TempInst`] instance has been modified, which is
    /// not possible via its API but can be achieved by swapping it with another instance, e.g.
    /// using [`core::mem::swap`].
    ///
    /// Unfortunately, a regular panic is not sufficient in this case because it can be caught
    /// with [`std::panic::catch_unwind`], and a dangling [`TempInst`] reference can then be
    /// obtained from the closure passed to [`std::panic::catch_unwind`] (safely, because
    /// unfortunately [`std::panic::UnwindSafe`] is not an `unsafe` trait -- why?!!).
    ///
    /// The panic behavior can be changed via [`wrappers::set_modification_panic_fn`].
    ///
    /// # Remarks
    ///
    /// For many types, including [`TempRef`], `T::Mutable` is actually the same as `T::Shared`.
    /// This can be useful when combining mutable and shared references in a tuple. E.g.
    /// `T = (TempRefMut<U>, TempRef<V>)` represents `(&mut U, &V)`, and this is preserved by
    /// [`Self::new_wrapper_mut`], whereas [`Self::new_wrapper`] treats it as `(&U, &V)`.
    pub unsafe fn new_wrapper_mut(obj: T::Mutable<'_>) -> TempInstWrapperMut<T>
    where
        T: Clone + PartialEq,
    {
        TempInstWrapperMut::new(obj)
    }

    /// Converts an instance of `T::Mutable` into a wrapper that can return a pinned mutable
    /// [`TempInst`] reference. The client needs to pin the wrapper before using it, usually via
    /// [`core::pin::pin!`].
    ///
    /// Afterwards, an instance of `T::Mutable` can be recovered from the [`TempInst`] reference
    /// via [`Self::get_mut_pinned`].
    ///
    /// Note that only the [`TempInst`] reference is pinned; this is completely independent of
    /// whether `T::Mutable` is a pinned reference. E.g. `T` can be [`TempRefMut`] or
    /// [`TempRefPin`], and then [`Self::get_mut_pinned`] will return a mutable or pinned mutable
    /// reference accordingly.
    pub fn new_wrapper_pin(obj: T::Mutable<'_>) -> TempInstWrapperPin<T> {
        TempInstWrapperPin::new(obj)
    }

    /// Calls `f` with a mutable reference to a [`TempInst`] instance constructed from `obj`, and
    /// returns the value returned by `f`.
    ///
    /// This method is a simple (but safe) wrapper around [`Self::new_wrapper_mut`], which
    /// potentially has a slight overhead. If possible, use [`Self::new_wrapper_pin`] (or
    /// [`Self::call_with_pin`]) instead.
    ///
    /// # Panics
    ///
    /// Calls [`std::process::abort`] if `f` modifies the internal state of the object that was
    /// passed to it. See [`Self::new_wrapper_mut`] for more information.
    pub fn call_with_mut<R>(obj: T::Mutable<'_>, f: impl FnOnce(&mut Self) -> R) -> R
    where
        T: Clone + PartialEq,
    {
        // SAFETY: Trivial because the scope of `inst` ends at the end of this function.
        let mut inst = unsafe { Self::new_wrapper_mut(obj) };
        f(&mut inst)
    }

    /// Calls `f` with a pinned mutable reference to a [`TempInst`] instance constructed from `obj`,
    /// and returns the value returned by `f`.
    ///
    /// This method exists for consistency (with respect to [`Self::call_with_mut`]), but is
    /// actually just a trivial application of [`Self::new_wrapper_pin`].
    pub fn call_with_pin<R>(obj: T::Mutable<'_>, f: impl FnOnce(Pin<&mut Self>) -> R) -> R {
        let inst = pin!(Self::new_wrapper_pin(obj));
        f(inst.deref_pin())
    }

    /// Returns the object that was originally passed to [`Self::new`], [`Self::new_wrapper_mut`],
    /// etc., with a lifetime that is restricted to that of `self`.
    pub fn get_mut(&mut self) -> T::Mutable<'_> {
        // SAFETY:
        //
        // Mutable instances of `TempInst<T>` can only be created via `new`, `new_wrapper_mut`, or
        // `new_wrapper_pin`, and all do so via `T::mut_to_temp`.
        //
        // `new` requires a static lifetime. `new_wrapper[...]` accepts any lifetime, but the
        // returned wrapper has the same lifetime, effectively borrowing all contained references
        // until the wrapper is dropped. Dereferencing the wrapper borrows it, so we can be sure
        // that the lifetime of the `TempInst` reference is outlived by the lifetime originally
        // passed to `mut_to_temp`, and that no other borrow can exist.
        //
        // A caveat is that mutable `TempInst` references can be passed to `core::mem::swap` and
        // related functions, counteracting the rules specified in `TempReprMut`. The swapping
        // operation itself is not problematic because it merely switches the role of the two
        // references with respect to all safety rules. However, the rules are violated when one of
        // the original borrows goes out of scope. By requiring that the `Drop` implementation of
        // the corresponding wrapper is executed, we make sure that the swap is detected at that
        // point.
        unsafe { self.0.temp_to_mut() }
    }

    /// Like [`Self::get_mut`], but accepts a pinned reference as created by
    /// [`Self::new_wrapper_pin`].
    pub fn get_mut_pinned(self: Pin<&mut Self>) -> T::Mutable<'_>
    where
        T: Unpin,
    {
        // SAFETY: Like `get_mut`, but pinning prevents clients from swapping instances.
        unsafe {
            self.map_unchecked_mut(|inst| &mut inst.0)
                .get_mut()
                .temp_to_mut()
        }
    }
}

impl<T: TempReprMut + 'static> Default for TempInst<T>
where
    T::Mutable<'static>: Default,
{
    fn default() -> Self {
        TempInst::new(T::Mutable::default())
    }
}

impl<T: TempRepr> PartialEq for TempInst<T>
where
    for<'a> T::Shared<'a>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: TempRepr> Eq for TempInst<T> where for<'a> T::Shared<'a>: Eq {}

impl<T: TempRepr> PartialOrd for TempInst<T>
where
    for<'a> T::Shared<'a>: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }
}

impl<T: TempRepr> Ord for TempInst<T>
where
    for<'a> T::Shared<'a>: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.get().cmp(&other.get())
    }
}

impl<T: TempRepr> Hash for TempInst<T>
where
    for<'a> T::Shared<'a>: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get().hash(state);
    }
}

impl<T: TempRepr> Debug for TempInst<T>
where
    for<'a> T::Shared<'a>: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

// SAFETY: See the condition at `TempRepr`.
unsafe impl<T: TempRepr> Sync for TempInst<T> where for<'a> T::Shared<'a>: Send {}

// SAFETY: This is only relevant for `TempInst` instances created with `new`. See the condition at
// `TempReprMut`.
unsafe impl<T: TempReprMut> Send for TempInst<T> where for<'a> T::Mutable<'a>: Send {}

pub mod wrappers {
    use super::*;

    /// A lifetime-dependent wrapper that contains a [`TempInst`] and hands out shared references to
    /// it via [`Deref`].
    pub struct TempInstWrapper<'a, T: TempRepr + 'a> {
        inst: TempInst<T>,
        phantom: PhantomData<T::Shared<'a>>,
    }

    impl<'a, T: TempRepr> TempInstWrapper<'a, T> {
        pub(crate) fn new(obj: T::Shared<'a>) -> Self {
            TempInstWrapper {
                inst: TempInst(T::shared_to_temp(obj), PhantomPinned),
                phantom: PhantomData,
            }
        }
    }

    impl<T: TempRepr> Deref for TempInstWrapper<'_, T> {
        type Target = TempInst<T>;

        fn deref(&self) -> &Self::Target {
            &self.inst
        }
    }

    /// A lifetime-dependent wrapper that contains a [`TempInst`] and hands out mutable references
    /// to it via [`DerefMut`].
    pub struct TempInstWrapperMut<'a, T: TempReprMut + Clone + PartialEq + 'a> {
        inst: TempInst<T>,
        orig: T,
        phantom: PhantomData<T::Mutable<'a>>,
    }

    impl<'a, T: TempReprMut + Clone + PartialEq> TempInstWrapperMut<'a, T> {
        // Safety: see `TempInst::new_wrapper_mut`.
        pub(crate) unsafe fn new(obj: T::Mutable<'a>) -> Self {
            let orig = T::mut_to_temp(obj);
            let inst = TempInst(orig.clone(), PhantomPinned);
            TempInstWrapperMut {
                inst,
                orig,
                phantom: PhantomData,
            }
        }
    }

    impl<T: TempReprMut + Clone + PartialEq> Drop for TempInstWrapperMut<'_, T> {
        fn drop(&mut self) {
            if self.inst.0 != self.orig {
                modification_panic();
            }
        }
    }

    impl<T: TempReprMut + Clone + PartialEq> Deref for TempInstWrapperMut<'_, T> {
        type Target = TempInst<T>;

        fn deref(&self) -> &Self::Target {
            &self.inst
        }
    }

    impl<T: TempReprMut + Clone + PartialEq> DerefMut for TempInstWrapperMut<'_, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.inst
        }
    }

    /// A lifetime-dependent wrapper that contains a [`TempInst`] and hands out pinned mutable
    /// references to it.
    pub struct TempInstWrapperPin<'a, T: TempReprMut + 'a> {
        inst: TempInst<T>,
        phantom: PhantomData<T::Mutable<'a>>,
    }

    impl<'a, T: TempReprMut> TempInstWrapperPin<'a, T> {
        pub(crate) fn new(obj: T::Mutable<'a>) -> Self {
            TempInstWrapperPin {
                inst: TempInst(T::mut_to_temp(obj), PhantomPinned),
                phantom: PhantomData,
            }
        }
    }

    impl<T: TempReprMut> Deref for TempInstWrapperPin<'_, T> {
        type Target = TempInst<T>;

        fn deref(&self) -> &Self::Target {
            &self.inst
        }
    }

    impl<T: TempReprMut> TempInstWrapperPin<'_, T> {
        /// Returns a pinned mutable [`TempInst`] reference, which can be used with
        /// [`TempInst::get_mut_pinned`].
        pub fn deref_pin(self: Pin<&mut Self>) -> Pin<&mut <Self as Deref>::Target> {
            unsafe { self.map_unchecked_mut(|p| &mut p.inst) }
        }
    }

    #[cfg(feature = "std")]
    fn modification_panic_fn() {
        // Note: Can be replaced by `std::panic::always_abort` once that is stabilized.
        let orig_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            orig_hook(panic_info);
            std::process::abort()
        }));
        panic!("TempInst instance was modified; this is not allowed because it violates safety guarantees");
    }

    #[cfg(not(feature = "std"))]
    fn modification_panic_fn() {
        // In the nostd case, we don't have `std::process::abort()`, so entering an endless loop
        // seems like the best we can do.
        loop {}
    }

    static mut MODIFICATION_PANIC_FN: fn() = modification_panic_fn;

    /// Sets an alternative function to be called by the wrapper returned by
    /// [`TempInst::new_wrapper_mut`] when it encounters an illegal modification.
    ///
    /// # Safety
    ///
    /// * Changing the panic function is only allowed when no other thread is able to interact with
    ///   this crate.
    ///
    /// * The panic function must not return unless [`std::thread::panicking`] returns `true`.
    ///
    /// * If the panic function causes an unwinding panic, the caller of
    ///   [`set_modification_panic_fn`] assumes the responsibility that no mutable [`TempInst`]
    ///   reference is used across (i.e. captured in) [`std::panic::catch_unwind`].
    pub unsafe fn set_modification_panic_fn(panic_fn: fn()) {
        MODIFICATION_PANIC_FN = panic_fn;
    }

    fn modification_panic() {
        // SAFETY: The safety conditions of `set_modification_panic_fn` guarantee that no other
        // thread is concurrently modifying the static variable.
        unsafe { MODIFICATION_PANIC_FN() }
    }
}

/// A trait that specifies that a type is a "temporary representation" of another type, where that
/// other type can depend on a lifetime (via GADTs). The standard example is that a raw pointer can
/// be regarded as a temporary representation of a reference. The trait implementation for tuples
/// generalizes this to combinations of more than one pointer/reference, the trait implementation
/// for [`Option`] extends it to optional references, etc.
///
/// Every type implementing [`TempRepr`] can be used in [`TempInst`], which provides a safe API
/// around the temporary representation.
///
/// # Safety
///
/// * The implementation of the trait must ensure that `shared_to_temp<'a>` followed by
///   `temp_to_shared<'b>` cannot cause undefined behavior when `'a: 'b`. (This essentially
///   restricts `Shared<'a>` to types that are covariant in `'a`.)
///
/// * If `Shared` implements `Send`, then it must be valid to send and access instances of `Temp`
///   across threads, and call `temp_to_shared` in a different thread than the original
///   `shared_to_temp` call.
pub unsafe trait TempRepr {
    /// The type that `Self` is a temporary representation of. May contain shared references of
    /// lifetime `'a`.
    type Shared<'a>
    where
        Self: 'a;

    /// Converts the given object to its temporary representation.
    fn shared_to_temp(obj: Self::Shared<'_>) -> Self;

    /// Converts from a shared temporary reference back to the original type with a
    /// suitably-restricted lifetime.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the lifetime that was used in the call to `shared_to_temp` (or
    /// `mut_to_temp`) outlives the lifetime passed to `temp_to_shared`, and that `temp_to_mut` is
    /// not called with an overlapping lifetime if [`TempReprMut`] is also implemented.
    ///
    /// (Exception: see the caveat about [`PartialEq`] in the safety rules of [`TempReprMut`].)
    unsafe fn temp_to_shared(&self) -> Self::Shared<'_>;
}

/// An extension of [`TempRepr`] that adds support for mutable references.
///
/// If the represented type has no special "mutable" variant, the [`AlwaysShared`] marker trait can
/// be used to implement [`TempReprMut`] identically to [`TempRepr`].
///
/// # Safety
///
/// In addition to the requirements of [`TempRepr`], the implementation needs to ensure the
/// following properties.
///
/// * `mut_to_temp<'a>` followed by `temp_to_mut<'b>` must not cause undefined behavior when
///   `'a: 'b` and `'b` does not overlap with any other lifetime passed to `temp_to_shared` or
///   `temp_to_mut`.
///
/// * `mut_to_temp<'a>` followed by `temp_to_shared<'b>` must not cause undefined behavior when
///   `'a: 'b` and `'b` does not overlap with any lifetime passed to `temp_to_mut`.
///
/// * The type `Temp` must implement [`Clone`] and [`PartialEq`] in such a way that a call to
///   [`core::mem::swap`] with two `Temp` references is either detectable by cloning before and
///   comparing afterwards, or is harmless. Whenever a swapping operation is not detected by the
///   [`PartialEq`] implementation, the swapped instances of `Temp` must be interchangeable in terms
///   of all other conditions. In particular, in all above points, undefined behavior must also be
///   avoided if "`'a: 'b`" is weakened to "`'c: 'b` for some `'c` such that the result of
///   `mut_to_temp<'c>` compared equal to the result of `mut_to_temp<'a>`".
///
/// * If `Mutable` implements `Send`, then it must be valid to send and access instances of `Temp`
///   across threads, and call `temp_to_shared` or `temp_to_mut` in a different thread than the
///   original `mut_to_temp` call.
pub unsafe trait TempReprMut: TempRepr {
    /// The type that `Self` is a temporary representation of. May contain mutable references of
    /// lifetime `'a`.
    type Mutable<'a>
    where
        Self: 'a;

    /// Converts the given object to a temporary representation without a lifetime parameter.
    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self;

    /// Converts from a mutable reference to the temporary representation back to the original type,
    /// with a restricted lifetime.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the temporary representation was created using `mut_to_temp`
    /// (not `shared_to_temp`), that the lifetime that was used in the call to `mut_to_temp`
    /// outlives the lifetime passed to `temp_to_mut`, and that neither `temp_to_shared` nor
    /// `temp_to_mut` are called with an overlapping lifetime.
    ///
    /// (Exception: see the caveat about [`PartialEq`] in the safety rules of [`TempReprMut`].)
    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_>;
}

/// A marker trait that causes [`TempReprMut`] to be implemented identically to [`TempRepr`].
pub trait AlwaysShared: TempRepr {}

// SAFETY: The additional conditions of `TempReprMut` are trivially satisfied if the references
// returned by `temp_to_mut` are actually shared references.
unsafe impl<T: AlwaysShared> TempReprMut for T {
    type Mutable<'a> = Self::Shared<'a> where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        Self::shared_to_temp(obj)
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        self.temp_to_shared()
    }
}

/// A wrapper type that trivially implements [`TempRepr`]/[`TempReprMut`] for any `T: Clone` in such
/// a way that no lifetimes are erased.
#[derive(Clone, PartialEq)]
pub struct SelfRepr<T: Clone>(T);

// SAFETY: All conditions are trivially satisfied because the `temp_to_shared` implementation isn't
// actually unsafe.
unsafe impl<T: Clone> TempRepr for SelfRepr<T> {
    type Shared<'a> = T where Self: 'a;

    fn shared_to_temp(obj: T) -> Self {
        SelfRepr(obj)
    }

    unsafe fn temp_to_shared(&self) -> T {
        self.0.clone()
    }
}

impl<T: Clone> AlwaysShared for SelfRepr<T> {}

/// The canonical implementation of [`TempRepr`], representing a single shared reference.
///
/// This is not necessarily very useful on its own, but forms the basis of composition via tuples.
#[derive(Clone, PartialEq)]
pub struct TempRef<T: ?Sized>(NonNull<T>);

// SAFETY: The safety rules of `TempRepr` are canonically satisfied by conversions between
// shared references and pointers.
unsafe impl<T: ?Sized> TempRepr for TempRef<T> {
    type Shared<'a> = &'a T where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        TempRef(obj.into())
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        // SAFETY: The safety rules of `temp_to_shared` ensure that this call is valid.
        self.0.as_ref()
    }
}

impl<T: ?Sized> AlwaysShared for TempRef<T> {}

/// The canonical implementation of [`TempReprMut`], representing a single mutable reference.
///
/// This is not necessarily very useful on its own, but forms the basis of composition via tuples.
#[derive(Clone, PartialEq)]
pub struct TempRefMut<T: ?Sized>(NonNull<T>);

// SAFETY: The safety rules of `TempRepr` are canonically satisfied by conversions between
// shared references and pointers.
unsafe impl<T: ?Sized> TempRepr for TempRefMut<T> {
    type Shared<'a> = &'a T where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        TempRefMut(obj.into())
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        // SAFETY: The safety rules of `temp_to_shared` ensure that this call is valid.
        self.0.as_ref()
    }
}

// SAFETY: The safety rules of `TempReprMut` are canonically satisfied by conversions between
// mutable references and pointers.
unsafe impl<T: ?Sized> TempReprMut for TempRefMut<T> {
    type Mutable<'a> = &'a mut T where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        TempRefMut(obj.into())
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        // SAFETY: The safety rules of `temp_to_mut` ensure that this call is valid.
        self.0.as_mut()
    }
}

/// Similar to [`TempRefMut`], but represents a pinned mutable reference.
#[derive(Clone, PartialEq)]
pub struct TempRefPin<T: ?Sized>(NonNull<T>);

// SAFETY: The safety rules of `TempRepr` are canonically satisfied by conversions between
// shared references and pointers.
unsafe impl<T: ?Sized> TempRepr for TempRefPin<T> {
    type Shared<'a> = &'a T where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        TempRefPin(obj.into())
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        // SAFETY: The safety rules of `temp_to_shared` ensure that this call is valid.
        self.0.as_ref()
    }
}

// SAFETY: The safety rules of `TempReprMut` are canonically satisfied by conversions between
// mutable references and pointers.
unsafe impl<T: ?Sized> TempReprMut for TempRefPin<T> {
    type Mutable<'a> = Pin<&'a mut T> where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        // SAFETY: Converting a pinned reference to a pointer is obviously unproblematic as long as
        // we only convert it back to a pinned or shared reference.
        unsafe { TempRefPin(obj.get_unchecked_mut().into()) }
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        // SAFETY: The safety rules of `temp_to_mut` ensure that this call is valid.
        Pin::new_unchecked(self.0.as_mut())
    }
}

unsafe impl<T: TempRepr> TempRepr for Option<T> {
    type Shared<'a> = Option<T::Shared<'a>> where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        obj.map(T::shared_to_temp)
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        Some(self.as_ref()?.temp_to_shared())
    }
}

unsafe impl<T: TempReprMut> TempReprMut for Option<T> {
    type Mutable<'a> = Option<T::Mutable<'a>> where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        obj.map(T::mut_to_temp)
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        Some(self.as_mut()?.temp_to_mut())
    }
}

#[cfg(feature = "std")]
#[derive(Clone, PartialEq)]
pub enum TempCow<T: ?Sized + ToOwned<Owned: Clone>> {
    Borrowed(TempRef<T>),
    Owned(T::Owned),
}

#[cfg(feature = "std")]
unsafe impl<T: ?Sized + ToOwned<Owned: Clone>> TempRepr for TempCow<T> {
    type Shared<'a> = std::borrow::Cow<'a, T> where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        match obj {
            std::borrow::Cow::Borrowed(obj) => TempCow::Borrowed(TempRef::shared_to_temp(obj)),
            std::borrow::Cow::Owned(obj) => TempCow::Owned(obj),
        }
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        match self {
            TempCow::Borrowed(temp) => std::borrow::Cow::Borrowed(temp.temp_to_shared()),
            TempCow::Owned(temp) => std::borrow::Cow::Owned(temp.clone()),
        }
    }
}

#[cfg(feature = "std")]
impl<T: ?Sized + ToOwned<Owned: Clone>> AlwaysShared for TempCow<T> {}

macro_rules! impl_temp_repr_tuple {
    ($($idx:tt $T:ident),*) => {
        unsafe impl<$($T: TempRepr),*> TempRepr for ($($T,)*) {
            type Shared<'a> = ($($T::Shared<'a>,)*) where Self: 'a;

            #[allow(unused_variables)]
            fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
                ($($T::shared_to_temp(obj.$idx),)*)
            }

            unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
                ($(self.$idx.temp_to_shared(),)*)
            }
        }

        unsafe impl<$($T: TempReprMut),*> TempReprMut for ($($T,)*) {
            type Mutable<'a> = ($($T::Mutable<'a>,)*) where Self: 'a;

            #[allow(unused_variables)]
            fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
                ($($T::mut_to_temp(obj.$idx),)*)
            }

            unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
                ($(self.$idx.temp_to_mut(),)*)
            }
        }
    };
}

impl_temp_repr_tuple!();
impl_temp_repr_tuple!(0 T0);
impl_temp_repr_tuple!(0 T0, 1 T1);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7, 8 T8);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7, 8 T8, 9 T9);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7, 8 T8, 9 T9, 10 T10);
impl_temp_repr_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7, 8 T8, 9 T9, 10 T10, 11 T11);

#[cfg(feature = "either")]
unsafe impl<T0: TempRepr, T1: TempRepr> TempRepr for either::Either<T0, T1> {
    type Shared<'a> = either::Either<T0::Shared<'a>, T1::Shared<'a>> where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        match obj {
            either::Either::Left(obj0) => either::Either::Left(T0::shared_to_temp(obj0)),
            either::Either::Right(obj1) => either::Either::Right(T1::shared_to_temp(obj1)),
        }
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        match self {
            either::Either::Left(self0) => either::Either::Left(self0.temp_to_shared()),
            either::Either::Right(self1) => either::Either::Right(self1.temp_to_shared()),
        }
    }
}

#[cfg(feature = "either")]
unsafe impl<T0: TempReprMut, T1: TempReprMut> TempReprMut for either::Either<T0, T1> {
    type Mutable<'a> = either::Either<T0::Mutable<'a>, T1::Mutable<'a>> where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        match obj {
            either::Either::Left(obj0) => either::Either::Left(T0::mut_to_temp(obj0)),
            either::Either::Right(obj1) => either::Either::Right(T1::mut_to_temp(obj1)),
        }
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        match self {
            either::Either::Left(self0) => either::Either::Left(self0.temp_to_mut()),
            either::Either::Right(self1) => either::Either::Right(self1.temp_to_mut()),
        }
    }
}

unsafe impl<T: TempRepr> TempRepr for Range<T> {
    type Shared<'a> = Range<T::Shared<'a>> where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        T::shared_to_temp(obj.start)..T::shared_to_temp(obj.end)
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        self.start.temp_to_shared()..self.end.temp_to_shared()
    }
}

unsafe impl<T: TempReprMut> TempReprMut for Range<T> {
    type Mutable<'a> = Range<T::Mutable<'a>> where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        T::mut_to_temp(obj.start)..T::mut_to_temp(obj.end)
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        self.start.temp_to_mut()..self.end.temp_to_mut()
    }
}

unsafe impl<T: TempRepr> TempRepr for RangeFrom<T> {
    type Shared<'a> = RangeFrom<T::Shared<'a>> where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        T::shared_to_temp(obj.start)..
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        self.start.temp_to_shared()..
    }
}

unsafe impl<T: TempReprMut> TempReprMut for RangeFrom<T> {
    type Mutable<'a> = RangeFrom<T::Mutable<'a>> where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        T::mut_to_temp(obj.start)..
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        self.start.temp_to_mut()..
    }
}

unsafe impl<T: TempRepr> TempRepr for RangeTo<T> {
    type Shared<'a> = RangeTo<T::Shared<'a>> where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        ..T::shared_to_temp(obj.end)
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        ..self.end.temp_to_shared()
    }
}

unsafe impl<T: TempReprMut> TempReprMut for RangeTo<T> {
    type Mutable<'a> = RangeTo<T::Mutable<'a>> where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        ..T::mut_to_temp(obj.end)
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        ..self.end.temp_to_mut()
    }
}

unsafe impl TempRepr for RangeFull {
    type Shared<'a> = RangeFull where Self: 'a;

    fn shared_to_temp(_obj: RangeFull) -> Self {
        ..
    }

    unsafe fn temp_to_shared(&self) -> RangeFull {
        ..
    }
}

unsafe impl TempReprMut for RangeFull {
    type Mutable<'a> = RangeFull where Self: 'a;

    fn mut_to_temp(_obj: RangeFull) -> Self {
        ..
    }

    unsafe fn temp_to_mut(&mut self) -> RangeFull {
        ..
    }
}

pub mod mapped {
    use super::*;

    /// A safe helper trait for implementing the unsafe [`TempRepr`] and [`TempReprMut`] traits for
    /// a type, by mapping between the type and one with a temporary representation.
    ///
    /// The trait must be implemented on some arbitrary type `T`; by convention this should be the
    /// same as `Self::Mutable<'static>`. Then [`MappedTempRepr<T>`] can be used as the argument of
    /// [`TempInst`].
    pub trait HasTempRepr {
        type Temp: TempReprMut;

        type Shared<'a>
        where
            Self: 'a;

        type Mutable<'a>
        where
            Self: 'a;

        fn shared_to_mapped(obj: Self::Shared<'_>) -> <Self::Temp as TempRepr>::Shared<'_>;

        fn mapped_to_shared(mapped: <Self::Temp as TempRepr>::Shared<'_>) -> Self::Shared<'_>;

        fn mut_to_mapped(obj: Self::Mutable<'_>) -> <Self::Temp as TempReprMut>::Mutable<'_>;

        fn mapped_to_mut(mapped: <Self::Temp as TempReprMut>::Mutable<'_>) -> Self::Mutable<'_>;
    }

    /// See [`HasTempRepr`].
    #[derive(Clone, PartialEq)]
    pub struct MappedTempRepr<T: HasTempRepr>(T::Temp);

    unsafe impl<T: HasTempRepr> TempRepr for MappedTempRepr<T> {
        type Shared<'a> = T::Shared<'a> where Self: 'a;

        fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
            MappedTempRepr(T::Temp::shared_to_temp(T::shared_to_mapped(obj)))
        }

        unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
            T::mapped_to_shared(self.0.temp_to_shared())
        }
    }

    unsafe impl<T: HasTempRepr> TempReprMut for MappedTempRepr<T> {
        type Mutable<'a> = T::Mutable<'a> where Self: 'a;

        fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
            MappedTempRepr(T::Temp::mut_to_temp(T::mut_to_mapped(obj)))
        }

        unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
            T::mapped_to_mut(self.0.temp_to_mut())
        }
    }

    impl<T> HasTempRepr for slice::Iter<'static, T> {
        type Temp = TempRef<[T]>;

        type Shared<'a> = slice::Iter<'a, T> where Self: 'a;
        type Mutable<'a> = slice::Iter<'a, T> where Self: 'a;

        fn shared_to_mapped(obj: Self::Shared<'_>) -> <Self::Temp as TempRepr>::Shared<'_> {
            obj.as_slice()
        }

        fn mapped_to_shared(mapped: <Self::Temp as TempRepr>::Shared<'_>) -> Self::Shared<'_> {
            mapped.iter()
        }

        fn mut_to_mapped(obj: Self::Mutable<'_>) -> <Self::Temp as TempReprMut>::Mutable<'_> {
            obj.as_slice()
        }

        fn mapped_to_mut(mapped: <Self::Temp as TempReprMut>::Mutable<'_>) -> Self::Mutable<'_> {
            mapped.iter()
        }
    }

    impl<T> HasTempRepr for slice::IterMut<'static, T> {
        type Temp = TempRefMut<[T]>;

        type Shared<'a> = slice::Iter<'a, T> where Self: 'a;
        type Mutable<'a> = slice::IterMut<'a, T> where Self: 'a;

        fn shared_to_mapped(obj: Self::Shared<'_>) -> <Self::Temp as TempRepr>::Shared<'_> {
            obj.as_slice()
        }

        fn mapped_to_shared(mapped: <Self::Temp as TempRepr>::Shared<'_>) -> Self::Shared<'_> {
            mapped.iter()
        }

        fn mut_to_mapped(obj: Self::Mutable<'_>) -> <Self::Temp as TempReprMut>::Mutable<'_> {
            obj.into_slice()
        }

        fn mapped_to_mut(mapped: <Self::Temp as TempReprMut>::Mutable<'_>) -> Self::Mutable<'_> {
            mapped.iter_mut()
        }
    }

    impl HasTempRepr for Chars<'static> {
        type Temp = TempRef<str>;

        type Shared<'a> = Chars<'a> where Self: 'a;
        type Mutable<'a> = Chars<'a> where Self: 'a;

        fn shared_to_mapped(obj: Self::Shared<'_>) -> <Self::Temp as TempRepr>::Shared<'_> {
            obj.as_str()
        }

        fn mapped_to_shared(mapped: <Self::Temp as TempRepr>::Shared<'_>) -> Self::Shared<'_> {
            mapped.chars()
        }

        fn mut_to_mapped(obj: Self::Mutable<'_>) -> <Self::Temp as TempReprMut>::Mutable<'_> {
            obj.as_str()
        }

        fn mapped_to_mut(mapped: <Self::Temp as TempReprMut>::Mutable<'_>) -> Self::Mutable<'_> {
            mapped.chars()
        }
    }
}

pub type TempSliceIter<T> = MappedTempRepr<slice::Iter<'static, T>>;
pub type TempSliceIterMut<T> = MappedTempRepr<slice::IterMut<'static, T>>;

pub type TempChars = MappedTempRepr<Chars<'static>>;

#[cfg(test)]
mod tests {
    use core::mem::swap;

    use super::*;

    #[cfg(feature = "std")]
    fn init_tests() {
        // Set up a non-aborting panic handler so we can test the panicking code.
        // We need to make sure `set_modification_panic_fn` is called before any other test code,
        // to satisfy its safety requirement.
        static INIT: std::sync::LazyLock<()> = std::sync::LazyLock::new(|| unsafe {
            set_modification_panic_fn(|| {
                if !std::thread::panicking() {
                    panic!("TempInst was modified");
                }
            })
        });
        *INIT
    }

    #[cfg(not(feature = "std"))]
    fn init_tests() {}

    #[test]
    fn temp_ref() {
        init_tests();
        let mut a = 42;
        let a_inst = TempInst::<TempRef<i32>>::new_wrapper(&a);
        let a_ref = a_inst.get();
        assert_eq!(*a_ref, 42);
        let double = 2 * *a_ref;
        assert_eq!(a, 42);
        a += 1; // Important: fails to compile if `a_inst` or `a_ref` are used afterwards.
        assert_eq!(a, 43);
        assert_eq!(double, 2 * 42);
    }

    #[test]
    fn temp_ref_call() {
        init_tests();
        let a = 42;
        let double = TempInst::<TempRef<i32>>::call_with(&a, |inst| {
            let a_ref = inst.get();
            assert_eq!(*a_ref, 42);
            2 * *a_ref
        });
        assert_eq!(a, 42);
        assert_eq!(double, 2 * 42);
    }

    #[test]
    fn temp_ref_call_pair() {
        init_tests();
        let a = 42;
        let b = 23;
        let sum = TempInst::<(TempRef<i32>, TempRef<i32>)>::call_with((&a, &b), |inst| {
            let (a_ref, b_ref) = inst.get();
            assert_eq!(*a_ref, 42);
            assert_eq!(*b_ref, 23);
            *a_ref + *b_ref
        });
        assert_eq!(a, 42);
        assert_eq!(b, 23);
        assert_eq!(sum, 42 + 23);
    }

    #[test]
    fn temp_ref_call_mut() {
        init_tests();
        let mut a = 42;
        let double = TempInst::<TempRefMut<i32>>::call_with_mut(&mut a, |inst| {
            let a_ref = inst.get_mut();
            assert_eq!(*a_ref, 42);
            *a_ref += 1;
            2 * *a_ref
        });
        assert_eq!(a, 43);
        assert_eq!(double, 2 * 43);
    }

    #[test]
    fn temp_ref_call_pair_mut() {
        init_tests();
        let mut a = 42;
        let mut b = 23;
        let sum = TempInst::<(TempRefMut<i32>, TempRefMut<i32>)>::call_with_mut(
            (&mut a, &mut b),
            |inst| {
                let (a_ref, b_ref) = inst.get_mut();
                assert_eq!(*a_ref, 42);
                assert_eq!(*b_ref, 23);
                *a_ref += 1;
                *b_ref -= 2;
                *a_ref + *b_ref
            },
        );
        assert_eq!(a, 43);
        assert_eq!(b, 21);
        assert_eq!(sum, 43 + 21);
    }

    #[test]
    fn temp_ref_call_pair_half_mut() {
        init_tests();
        let mut a = 42;
        let b = 23;
        let sum =
            TempInst::<(TempRefMut<i32>, TempRef<i32>)>::call_with_mut((&mut a, &b), |inst| {
                let (a_ref, b_ref) = inst.get_mut();
                assert_eq!(*a_ref, 42);
                assert_eq!(*b_ref, 23);
                *a_ref += 1;
                *a_ref + *b_ref
            });
        assert_eq!(a, 43);
        assert_eq!(b, 23);
        assert_eq!(sum, 43 + 23);
    }

    #[test]
    fn temp_ref_call_pin() {
        init_tests();
        let mut a = 42;
        let double = TempInst::<TempRefMut<i32>>::call_with_pin(&mut a, |inst| {
            let a_ref = inst.get_mut_pinned();
            assert_eq!(*a_ref, 42);
            *a_ref += 1;
            2 * *a_ref
        });
        assert_eq!(a, 43);
        assert_eq!(double, 2 * 43);
    }

    #[cfg(feature = "std")]
    #[test]
    #[should_panic]
    fn temp_ref_call_mut_illegal_swap() {
        init_tests();
        let mut a = 42;
        TempInst::<TempRefMut<i32>>::call_with_mut(&mut a, |a_inst| {
            let mut b = 43;
            TempInst::<TempRefMut<i32>>::call_with_mut(&mut b, |b_inst| {
                swap(a_inst, b_inst);
            });
            // This would cause undefined behavior due to swapping, if undetected.
            let b_ref = a_inst.get_mut();
            assert_ne!(*b_ref, 43);
        });
    }

    #[test]
    fn temp_ref_call_mut_swap_zero_size() {
        // The zero-size case is special because both halves of each pair have the same address in
        // memory, so the `swap` operation is undetectable. This is hopefully not a problem because
        // we can consider it as a no-op then. In particular, we can hopefully assume that both
        // lifetimes end at the same time, and that the references are completely interchangeable
        // until then.
        init_tests();
        let mut a = ((), ());
        TempInst::<TempRefMut<()>>::call_with_mut(&mut a.0, |a0_inst| {
            TempInst::<TempRefMut<()>>::call_with_mut(&mut a.1, |a1_inst| {
                swap(a0_inst, a1_inst);
            });
            // This produces multiple mutable references to a.1, which could potentially be
            // regarded as causing undefined behavior. Miri is OK with it, though.
            let a1_ref = &mut a.1;
            let a1_ref_2 = a0_inst.get_mut();
            *a1_ref = ();
            *a1_ref_2 = ();
        });
    }
}
