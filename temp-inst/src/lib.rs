#![cfg_attr(not(feature = "std"), no_std)]

use core::{
    cmp::Ordering,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::{PhantomData, PhantomPinned},
    ops::{Deref, DerefMut, Range, RangeFrom, RangeFull, RangeTo},
    pin::{pin, Pin},
    ptr::{self, NonNull},
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
/// // We want to implement this example trait for a specific type `Bar`, in order to call
/// // `run_twice` below.
/// pub trait Foo {
///     type Arg;
///
///     fn run(arg: &mut Self::Arg);
/// }
///
/// pub fn run_twice<F: Foo>(arg: &mut F::Arg) {
///     F::run(arg);
///     F::run(arg);
/// }
///
/// struct Bar;
///
/// impl Foo for Bar {
///     // We actually want to use _two_ mutable references as the argument type. However, the
///     // associated type `Arg` does not have any lifetime parameter. If we can add a lifetime
///     // parameter `'a` to `Bar`, then `type Arg = (&'a mut i32, &'a mut i32)` will work. If we
///     // can't or don't want to do that, an equivalent `TempInst` will do the trick.
///     type Arg = TempInst<(TempRefMut<i32>, TempRefMut<i32>)>;
///
///     fn run(arg: &mut Self::Arg) {
///         // From a mutable `TempInst` reference, we can extract the mutable references that we
///         // originally constructed it from.
///         let (a_ref, b_ref) = arg.get_mut();
///         *a_ref += *b_ref;
///         *b_ref += 1;
///     }
/// }
///
/// let mut a = 42;
/// let mut b = 23;
///
/// // Now we can convert the pair `(&mut a, &mut b)` to a mutable `TempInst` reference, and pass
/// // that to `run_twice`.
/// TempInst::call_with_mut((&mut a, &mut b), run_twice::<Bar>);
///
/// assert_eq!(a, 42 + 23 + 1 + 23);
/// assert_eq!(b, 23 + 1 + 1);
/// ```
///
/// For shared or pinned mutable references, there is a slightly simpler API:
///
/// ```
/// # use crate::temp_inst::*;
///
/// pub trait Foo {
///     type Arg;
///
///     fn run(arg: &Self::Arg) -> i32;
/// }
///
/// fn run_twice_and_add<F: Foo>(arg: &F::Arg) -> i32 {
///     F::run(arg) + F::run(arg)
/// }
///
/// struct Bar;
///
/// impl Foo for Bar {
///     type Arg = TempInst<(TempRef<i32>, TempRef<i32>)>;
///
///     fn run(arg: &Self::Arg) -> i32 {
///         let (a_ref, b_ref) = arg.get();
///         *a_ref * *b_ref
///     }
/// }
///
/// let inst = TempInst::new_wrapper((&42, &23));
/// let sum_of_products = run_twice_and_add::<Bar>(&inst);
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
    #[must_use]
    pub fn new_wrapper(obj: T::Shared<'_>) -> TempInstWrapper<'_, T> {
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
    #[must_use]
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
    #[must_use]
    pub fn new(obj: T::Mutable<'static>) -> Self {
        TempInst(T::mut_to_temp(obj), PhantomPinned)
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
    #[must_use]
    pub fn new_wrapper_pin(obj: T::Mutable<'_>) -> TempInstWrapperPin<'_, T> {
        TempInstWrapperPin::new(obj)
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

    /// Like [`Self::get_mut`], but accepts a pinned reference as created by
    /// [`Self::new_wrapper_pin`].
    #[must_use]
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

impl<T: TempReprMutCmp> TempInst<T> {
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
    #[must_use]
    pub unsafe fn new_wrapper_mut(obj: T::Mutable<'_>) -> TempInstWrapperMut<'_, T> {
        TempInstWrapperMut::new(obj)
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
    pub fn call_with_mut<R>(obj: T::Mutable<'_>, f: impl FnOnce(&mut Self) -> R) -> R {
        // SAFETY: Trivial because the scope of `inst` ends at the end of this function.
        let mut inst = unsafe { Self::new_wrapper_mut(obj) };
        f(&mut inst)
    }

    /// Returns the object that was originally passed to [`Self::new`], [`Self::new_wrapper_mut`],
    /// etc., with a lifetime that is restricted to that of `self`.
    #[must_use]
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

pub mod wrappers {
    use super::*;

    /// A lifetime-dependent wrapper that contains a [`TempInst`] and hands out shared references to
    /// it via [`Deref`].
    pub struct TempInstWrapper<'a, T: TempRepr + 'a> {
        inst: TempInst<T>,
        phantom: PhantomData<T::Shared<'a>>,
    }

    impl<'a, T: TempRepr> TempInstWrapper<'a, T> {
        #[must_use]
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

    /// A lifetime-dependent wrapper that contains a [`TempInst`] and hands out pinned mutable
    /// references to it.
    pub struct TempInstWrapperPin<'a, T: TempReprMut + 'a> {
        inst: TempInst<T>,
        phantom: PhantomData<T::Mutable<'a>>,
    }

    impl<'a, T: TempReprMut> TempInstWrapperPin<'a, T> {
        #[must_use]
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
        #[must_use]
        pub fn deref_pin(self: Pin<&mut Self>) -> Pin<&mut <Self as Deref>::Target> {
            unsafe { self.map_unchecked_mut(|p| &mut p.inst) }
        }
    }

    /// A lifetime-dependent wrapper that contains a [`TempInst`] and hands out mutable references
    /// to it via [`DerefMut`].
    pub struct TempInstWrapperMut<'a, T: TempReprMutCmp + 'a> {
        inst: TempInst<T>,
        orig: T,
        phantom: PhantomData<T::Mutable<'a>>,
    }

    impl<'a, T: TempReprMutCmp> TempInstWrapperMut<'a, T> {
        // Safety: see `TempInst::new_wrapper_mut`.
        #[must_use]
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

    impl<T: TempReprMutCmp> Drop for TempInstWrapperMut<'_, T> {
        fn drop(&mut self) {
            if self.inst.0 != self.orig {
                modification_panic();
            }
        }
    }

    impl<T: TempReprMutCmp> Deref for TempInstWrapperMut<'_, T> {
        type Target = TempInst<T>;

        fn deref(&self) -> &Self::Target {
            &self.inst
        }
    }

    impl<T: TempReprMutCmp> DerefMut for TempInstWrapperMut<'_, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.inst
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
        // seems to be the best we can do.
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
/// other type can depend on a lifetime (via GATs). The standard example is that a raw pointer can
/// be regarded as a temporary representation of a reference. The trait implementation for tuples
/// generalizes this to combinations of more than one pointer/reference, the trait implementation
/// for [`Option`] extends it to optional references, etc.
///
/// Every type implementing [`TempRepr`] can be used in [`TempInst`], which provides a safe API
/// around the temporary representation.
///
/// Rather than implementing this trait directly, it is recommended to do so by defining a mapping
/// to and from built-in types, using the safe trait [`mapped::HasTempRepr`].
///
/// # Safety
///
/// * The implementation of the trait must ensure that `shared_to_temp<'a>` followed by
///   `temp_to_shared<'b>` cannot cause undefined behavior when `'a: 'b`. (This essentially
///   restricts `Shared<'a>` to types that are covariant in `'a`.)
///
/// * The above must also hold if a (legal) cast was applied to the result of `shared_to_temp`.
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
    /// (Exception: see the caveat about [`PartialEq`] in the safety rules of [`TempReprMutCmp`].)
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
/// * The above must also hold if a (legal) cast was applied to the result of `mut_to_temp`.
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
    /// (Exception: see the caveat about [`PartialEq`] in the safety rules of [`TempReprMutCmp`].)
    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_>;
}

/// An extension of [`TempReprMut`] that allows mutable references to be passed to safe client code.
///
/// # Safety
///
/// [`Clone`] and [`PartialEq`] must be implemented in such a way that a call to [`core::mem::swap`]
/// is either detectable by cloning before and comparing afterwards, or is harmless. Whenever a
/// swapping operation is not detected by the [`PartialEq`] implementation, the swapped instances
/// must be interchangeable in terms of all safety conditions of [`TempRepr`] and [`TempReprMut`].
/// In particular, in all specific points, undefined behavior must also be avoided when the
/// condition "`'a: 'b`" is weakened to "`'c: 'b` for some `'c` such that the result of
/// `shared_to_temp<'c>`/`mut_to_temp<'c>` compared equal to the result of
/// `shared_to_temp<'a>`/`mut_to_temp<'a>`".
pub unsafe trait TempReprMutCmp: TempReprMut + Clone + PartialEq {}

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

/*******************
 * SelfRepr<T> [T] *
 *******************/

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

// SAFETY: Trivially satisfied because the `temp_to_shared` implementation isn't actually unsafe.
unsafe impl<T: Clone + PartialEq> TempReprMutCmp for SelfRepr<T> {}

/*******************
 * TempRef<T> [&T] *
 *******************/

/// The canonical implementation of [`TempRepr`], representing a single shared reference.
///
/// This is not necessarily very useful on its own, but forms the basis of composition via tuples.
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

impl<T: ?Sized> Clone for TempRef<T> {
    fn clone(&self) -> Self {
        TempRef(self.0)
    }
}

impl<T: ?Sized> PartialEq for TempRef<T> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

// SAFETY: Equal pointers to the same type must point to the same object, unless the object is
// zero-size.
unsafe impl<T: ?Sized> TempReprMutCmp for TempRef<T> {}

// SAFETY: `TempRef<T>` follows the same rules as `&T` regarding thread safety.
unsafe impl<T: ?Sized + Sync> Send for TempRef<T> {}
unsafe impl<T: ?Sized + Sync> Sync for TempRef<T> {}

/**************************
 * TempRefMut<T> [&mut T] *
 **************************/

/// The canonical implementation of [`TempReprMut`], representing a single mutable reference.
///
/// This is not necessarily very useful on its own, but forms the basis of composition via tuples.
pub struct TempRefMut<T: ?Sized>(NonNull<T>, PhantomData<fn(T) -> T>);

// SAFETY: The safety rules of `TempRepr` are canonically satisfied by conversions between
// shared references and pointers.
unsafe impl<T: ?Sized> TempRepr for TempRefMut<T> {
    type Shared<'a> = &'a T where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        TempRefMut(obj.into(), PhantomData)
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        // SAFETY: The safety rules of `temp_to_shared` ensure that this call is valid.
        self.0.as_ref()
    }
}

// SAFETY: The safety rules of `TempReprMut` are canonically satisfied by conversions between
// mutable references and pointers.
//
// The `PhantomData` field guarantees that a `TempRefMut` instance cannot be cast in a covariant
// way, which `NonNull` would allow but violates the safety rules of `TempReprMut`.
unsafe impl<T: ?Sized> TempReprMut for TempRefMut<T> {
    type Mutable<'a> = &'a mut T where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        TempRefMut(obj.into(), PhantomData)
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        // SAFETY: The safety rules of `temp_to_mut` ensure that this call is valid.
        self.0.as_mut()
    }
}

impl<T: ?Sized> Clone for TempRefMut<T> {
    fn clone(&self) -> Self {
        TempRefMut(self.0, self.1)
    }
}

impl<T: ?Sized> PartialEq for TempRefMut<T> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

// SAFETY: Equal pointers to the same type must point to the same object, unless the object is
// zero-size.
unsafe impl<T: ?Sized> TempReprMutCmp for TempRefMut<T> {}

// SAFETY: `TempRefMut<T>` follows the same rules as `&mut T` regarding thread safety.
unsafe impl<T: ?Sized + Send> Send for TempRefMut<T> {}
unsafe impl<T: ?Sized + Sync> Sync for TempRefMut<T> {}

/*******************************
 * TempRefPin<T> [Pin<&mut T>] *
 *******************************/

/// Similar to [`TempRefMut`], but represents a pinned mutable reference.
pub struct TempRefPin<T: ?Sized>(NonNull<T>, PhantomData<fn(T) -> T>);

// SAFETY: The safety rules of `TempRepr` are canonically satisfied by conversions between
// shared references and pointers.
unsafe impl<T: ?Sized> TempRepr for TempRefPin<T> {
    type Shared<'a> = &'a T where Self: 'a;

    fn shared_to_temp(obj: Self::Shared<'_>) -> Self {
        TempRefPin(obj.into(), PhantomData)
    }

    unsafe fn temp_to_shared(&self) -> Self::Shared<'_> {
        // SAFETY: The safety rules of `temp_to_shared` ensure that this call is valid.
        self.0.as_ref()
    }
}

// SAFETY: The safety rules of `TempReprMut` are canonically satisfied by conversions between
// mutable references and pointers, and `Pin<Ptr>` is covariant in `Ptr`.
//
// The `PhantomData` field guarantees that a `TempRefPin` instance cannot be cast in a covariant
// way, which `NonNull` would allow but violates the safety rules of `TempReprMut`.
unsafe impl<T: ?Sized> TempReprMut for TempRefPin<T> {
    type Mutable<'a> = Pin<&'a mut T> where Self: 'a;

    fn mut_to_temp(obj: Self::Mutable<'_>) -> Self {
        // SAFETY: Converting a pinned reference to a pointer is obviously unproblematic as long as
        // we only convert it back to a pinned or shared reference.
        unsafe { TempRefPin(obj.get_unchecked_mut().into(), PhantomData) }
    }

    unsafe fn temp_to_mut(&mut self) -> Self::Mutable<'_> {
        // SAFETY: The safety rules of `temp_to_mut` ensure that this call is valid.
        Pin::new_unchecked(self.0.as_mut())
    }
}

impl<T: ?Sized> Clone for TempRefPin<T> {
    fn clone(&self) -> Self {
        TempRefPin(self.0, self.1)
    }
}

impl<T: ?Sized> PartialEq for TempRefPin<T> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

// SAFETY: Equal pointers to the same type must point to the same object, unless the object is
// zero-size.
unsafe impl<T: ?Sized> TempReprMutCmp for TempRefPin<T> {}

// SAFETY: `TempRefPin<T>` follows the same rules as `Pin<&mut T>` regarding thread safety.
unsafe impl<T: ?Sized + Send> Send for TempRefPin<T> {}
unsafe impl<T: ?Sized + Sync> Sync for TempRefPin<T> {}

/***********************
 * TempCow<T> [Cow<T>] *
 ***********************/

#[cfg(feature = "std")]
pub enum TempCow<T: ?Sized + ToOwned> {
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

#[cfg(feature = "std")]
impl<T: ?Sized + ToOwned<Owned: Clone>> Clone for TempCow<T> {
    fn clone(&self) -> Self {
        match self {
            TempCow::Borrowed(temp) => TempCow::Borrowed(temp.clone()),
            TempCow::Owned(temp) => TempCow::Owned(temp.clone()),
        }
    }
}

#[cfg(feature = "std")]
impl<T: ?Sized + ToOwned<Owned: Clone + PartialEq>> PartialEq for TempCow<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TempCow::Borrowed(l0), TempCow::Borrowed(r0)) => l0 == r0,
            (TempCow::Owned(l0), TempCow::Owned(r0)) => l0 == r0,
            _ => false,
        }
    }
}

// SAFETY: Note that we only care about the case where both instances are borrowed. In particular,
// `eq` never returns `true` when one instance is borrowed and the other is owned.
#[cfg(feature = "std")]
unsafe impl<T: ?Sized + ToOwned<Owned: Clone + PartialEq>> TempReprMutCmp for TempCow<T> {}

/*************
 * Option<T> *
 *************/

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

unsafe impl<T: TempReprMutCmp> TempReprMutCmp for Option<T> {}

/*****************
 * (T0, T1, ...) *
 *****************/

macro_rules! impl_temp_repr_tuple {
    ($($idx:tt $T:ident),*) => {
        #[allow(clippy::unused_unit)]
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

        #[allow(clippy::unused_unit)]
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

        unsafe impl<$($T: TempReprMutCmp),*> TempReprMutCmp for ($($T,)*) {}
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

/******************
 * Either<T0, T1> *
 ******************/

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

#[cfg(feature = "either")]
unsafe impl<T0: TempReprMutCmp, T1: TempReprMutCmp> TempReprMutCmp for either::Either<T0, T1> {}

/************
 * Range<T> *
 ************/

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

unsafe impl<T: TempReprMutCmp> TempReprMutCmp for Range<T> {}

/****************
 * RangeFrom<T> *
 ****************/

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

unsafe impl<T: TempReprMutCmp> TempReprMutCmp for RangeFrom<T> {}

/**************
 * RangeTo<T> *
 **************/

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

unsafe impl<T: TempReprMutCmp> TempReprMutCmp for RangeTo<T> {}

/*************
 * RangeFull *
 *************/

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

unsafe impl TempReprMutCmp for RangeFull {}

/****************
 * Mapped types *
 ****************/

pub mod mapped {
    use super::*;

    /// A safe helper trait for implementing the unsafe [`TempRepr`] and [`TempReprMut`] traits for
    /// a type, by mapping between the type and one with a temporary representation.
    ///
    /// The trait must be implemented on some arbitrary type `T`; by convention this should be the
    /// same as `Self::Mutable<'static>`. Then [`MappedTempRepr<T>`] can be used as the argument of
    /// [`TempInst`].
    pub trait HasTempRepr {
        type Temp: TempReprMutCmp;

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

    impl<T: HasTempRepr> Clone for MappedTempRepr<T> {
        fn clone(&self) -> Self {
            MappedTempRepr(self.0.clone())
        }
    }

    impl<T: HasTempRepr> PartialEq for MappedTempRepr<T> {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    unsafe impl<T: HasTempRepr> TempReprMutCmp for MappedTempRepr<T> {}

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
        let double = TempInst::<TempRef<i32>>::call_with(&a, |a_inst| {
            let a_ref = a_inst.get();
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
        let sum = TempInst::<(TempRef<i32>, TempRef<i32>)>::call_with((&a, &b), |a_b_inst| {
            let (a_ref, b_ref) = a_b_inst.get();
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
        let double = TempInst::<TempRefMut<i32>>::call_with_mut(&mut a, |a_inst| {
            let a_ref = a_inst.get_mut();
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
            |a_b_inst| {
                let (a_ref, b_ref) = a_b_inst.get_mut();
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
            TempInst::<(TempRefMut<i32>, TempRef<i32>)>::call_with_mut((&mut a, &b), |a_b_inst| {
                let (a_ref, b_ref) = a_b_inst.get_mut();
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
    fn temp_ref_mut_pin() {
        init_tests();
        let mut a = 42;
        let a_inst = pin!(TempInst::<TempRefMut<i32>>::new_wrapper_pin(&mut a));
        let a_ref = a_inst.deref_pin().get_mut_pinned();
        assert_eq!(*a_ref, 42);
        *a_ref += 1;
        let double = 2 * *a_ref;
        assert_eq!(a, 43);
        assert_eq!(double, 2 * 43);
    }

    #[test]
    fn temp_ref_mut_call_pin() {
        init_tests();
        let mut a = 42;
        let double = TempInst::<TempRefMut<i32>>::call_with_pin(&mut a, |a_inst| {
            let a_ref = a_inst.get_mut_pinned();
            assert_eq!(*a_ref, 42);
            *a_ref += 1;
            2 * *a_ref
        });
        assert_eq!(a, 43);
        assert_eq!(double, 2 * 43);
    }

    #[test]
    fn temp_ref_pin_call_pin() {
        init_tests();
        let mut a = pin!(42);
        let double = TempInst::<TempRefPin<i32>>::call_with_pin(Pin::as_mut(&mut a), |a_inst| {
            let a_ref = a_inst.get_mut_pinned().get_mut();
            assert_eq!(*a_ref, 42);
            *a_ref += 1;
            2 * *a_ref
        });
        assert_eq!(*a, 43);
        assert_eq!(double, 2 * 43);
    }

    #[cfg(feature = "std")]
    #[test]
    #[should_panic(expected = "TempInst was modified")]
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

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_send() {
        init_tests();
        let a = 42;
        let a_inst = TempInst::<TempRef<i32>>::new_wrapper(&a);
        std::thread::scope(|scope| {
            let thread = scope.spawn(move || *a_inst.get());
            let result = thread.join().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_sync() {
        init_tests();
        let a = 42;
        let a_inst = TempInst::<TempRef<i32>>::new_wrapper(&a);
        std::thread::scope(|scope| {
            let thread = scope.spawn(|| *a_inst.get());
            let result = thread.join().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_pin_send() {
        init_tests();
        let a = pin!(42);
        let a_inst = TempInst::<TempRefPin<i32>>::new_wrapper_pin(a);
        std::thread::scope(|scope| {
            let thread = scope.spawn(move || *a_inst.get());
            let result = thread.join().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_pin_sync() {
        init_tests();
        let a = pin!(42);
        let a_inst = TempInst::<TempRefPin<i32>>::new_wrapper_pin(a);
        std::thread::scope(|scope| {
            let thread = scope.spawn(|| *a_inst.get());
            let result = thread.join().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_call_mut_send() {
        init_tests();
        let mut a = 42;
        TempInst::<TempRefMut<i32>>::call_with_mut(&mut a, |a_inst| {
            std::thread::scope(|scope| {
                let thread = scope.spawn(move || *a_inst.get_mut() += 1);
                thread.join().unwrap();
            })
        });
        assert_eq!(a, 43);
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_call_mut_sync() {
        init_tests();
        let mut a = 42;
        TempInst::<TempRefMut<i32>>::call_with_mut(&mut a, |a_inst| {
            std::thread::scope(|scope| {
                let thread = scope.spawn(|| *a_inst.get_mut() += 1);
                thread.join().unwrap();
            })
        });
        assert_eq!(a, 43);
    }
}
