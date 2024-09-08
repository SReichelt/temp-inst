//! This crate provides safe lifetime-erased representations for objects with lifetime parameters.
//! Safety is achieved by making the lifetime-erased objects accessible only via short-lived
//! references.
//!
//! The main use case is to convert multiple (shared or mutable) references into a single reference
//! to a lifetime-erased object, which can then be passed to an API that only accepts a single
//! reference.
//!
//! A lifetime-erased object can be obtained by creating one of [`TempInst`], [`TempInstMut`], or
//! [`TempInstPin`], and dereferencing it. In the case of [`TempInstMut`], the
//! [`TempInstMut::call_with`] method should be used because [`TempInstMut::new`] is unsafe.
//!
//! To see which types have a temporary representation, see the implementations of the [`TempRepr`]
//! trait. It is possible to add temporary representations for custom types, preferably via
//! [`mapped::HasTempRepr`].
//!
//! # Examples
//!
//! ```
//! # use crate::temp_inst::*;
//!
//! // We want to implement this example trait for a specific type `Bar`, in order to call
//! // `run_twice` below.
//! pub trait Foo {
//!     type Arg;
//!
//!     fn run(arg: &mut Self::Arg);
//! }
//!
//! pub fn run_twice<F: Foo>(arg: &mut F::Arg) {
//!     F::run(arg);
//!     F::run(arg);
//! }
//!
//! struct Bar;
//!
//! impl Foo for Bar {
//!     // We actually want to use _two_ mutable references as the argument type. However, the
//!     // associated type `Arg` does not have any lifetime parameter. If we can add a lifetime
//!     // parameter `'a` to `Bar`, then `type Arg = (&'a mut i32, &'a mut i32)` will work. If we
//!     // can't or don't want to do that, a pair of `TempRefMut` will do the trick.
//!     type Arg = (TempRefMut<i32>, TempRefMut<i32>);
//!
//!     fn run(arg: &mut Self::Arg) {
//!         // The mutable `TempRefMut` references can be dereferenced to obtain the mutable
//!         // references that we passed to `call_with` below.
//!         let (a_ref, b_ref) = arg;
//!         **a_ref += **b_ref;
//!         **b_ref += 1;
//!     }
//! }
//!
//! let mut a = 42;
//! let mut b = 23;
//!
//! // Now we can convert the pair `(&mut a, &mut b)` to `&mut Foo::Arg`, and pass that to
//! // `run_twice`.
//! TempInstMut::call_with((&mut a, &mut b), run_twice::<Bar>);
//!
//! assert_eq!(a, 42 + 23 + 1 + 23);
//! assert_eq!(b, 23 + 1 + 1);
//! ```
//!
//! For shared or pinned mutable references, there is a slightly simpler API:
//!
//! ```
//! # use crate::temp_inst::*;
//!
//! pub trait Foo {
//!     type Arg;
//!
//!     fn run(arg: &Self::Arg) -> i32;
//! }
//!
//! fn run_twice_and_add<F: Foo>(arg: &F::Arg) -> i32 {
//!     F::run(arg) + F::run(arg)
//! }
//!
//! struct Bar;
//!
//! impl Foo for Bar {
//!     type Arg = (TempRef<i32>, TempRef<i32>);
//!
//!     fn run(arg: &Self::Arg) -> i32 {
//!         let (a_ref, b_ref) = arg;
//!         **a_ref * **b_ref
//!     }
//! }
//!
//! let inst = TempInst::new((&42, &23));
//! let sum_of_products = run_twice_and_add::<Bar>(&inst);
//!
//! assert_eq!(sum_of_products, 42 * 23 + 42 * 23);
//! ```
//!
//! For another use case, see [`temp_stack::TempStack`].

#![cfg_attr(not(feature = "std"), no_std)]

use core::{
    cmp::Ordering,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Deref, DerefMut, Range, RangeFrom, RangeFull, RangeTo},
    pin::{pin, Pin},
    ptr::NonNull,
    slice,
    str::Chars,
};

#[cfg(feature = "alloc")]
extern crate alloc;

use mapped::*;

pub mod temp_stack;

/// A wrapper around an instance of `T` which implements [`TempRepr`], i.e. is a temporary
/// representation of a type `T::Shared<'a>`.
///
/// [`TempInst`] itself has a lifetime parameter `'a` and borrows the object passed to
/// [`TempInst::new`] for that lifetime; therefore it is logically equivalent to `T::Shared<'a>`.
/// However, it can hand out references to the lifetime-less type `T` via its [`Deref`]
/// implementation.
///
/// See the module documentation for usage examples.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TempInst<'a, T: TempRepr + 'a>(T, PhantomData<T::Shared<'a>>);

impl<'a, T: TempRepr> TempInst<'a, T> {
    /// Creates a [`TempInst`] from an instance of `T::Shared`. Note that `T::Shared` is always the
    /// non-mutable variant of `T`; e.g. even if `T` is [`TempRefMut<X>`], `T::Shared` is `&X`, not
    /// `&mut X`.
    ///
    /// A shared reference to `T` can be obtained from the result via [`Deref`].
    #[must_use]
    pub fn new(obj: T::Shared<'a>) -> Self {
        // SAFETY: `obj` is borrowed for `'a`, which outlives the returned instance.
        unsafe { TempInst(T::new_temp(obj), PhantomData) }
    }

    /// Calls `f` with a shared reference to `T` which is constructed from `obj`, and returns the
    /// value returned by `f`.
    ///
    /// This method exists for consistency (with respect to [`TempInstMut::call_with`]), but is
    /// actually just a trivial application of [`Self::new`] and [`Deref::deref`].
    pub fn call_with<R>(obj: T::Shared<'a>, f: impl FnOnce(&T) -> R) -> R {
        let inst = Self::new(obj);
        f(&inst)
    }
}

impl<T: TempRepr> Deref for TempInst<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T: TempRepr<Shared<'a>: Default>> Default for TempInst<'a, T> {
    fn default() -> Self {
        TempInst::new(Default::default())
    }
}

impl<T: TempRepr + Debug> Debug for TempInst<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A wrapper around an instance of `T` which implements [`TempReprMut`], i.e. is a temporary
/// representation of a type `T::Mutable<'a>`.
///
/// [`TempInstPin`] itself has a lifetime parameter `'a` and borrows the object passed to
/// [`TempInstPin::new`] for that lifetime; therefore it is logically equivalent to
/// `T::Mutable<'a>`. However, it can hand out references to the lifetime-less type `T` via
/// [`TempInstPin::deref_pin`]. The references have type `Pin<&mut T>`; use the slightly less
/// efficient [`TempInstMut`] wrapper if `&mut T` is required instead.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TempInstPin<'a, T: TempReprMut + 'a>(T, PhantomData<T::Mutable<'a>>);

impl<'a, T: TempReprMut> TempInstPin<'a, T> {
    /// Creates a [`TempInstPin`] from an instance of `T::Mutable`.
    ///
    /// A pinned mutable reference to `T` can be obtained from the result via [`Self::deref_pin`].
    /// As `deref_pin` expects `self` to be pinned, use the [`core::pin::pin!`] macro to pin the
    /// result of [`Self::new`] on the stack.
    ///
    /// Note that only the [`TempInstPin`] reference will be pinned; this is completely independent
    /// of whether `T::Mutable` is a pinned reference. E.g. `T` can be [`TempRefMut`] or
    /// [`TempRefPin`], and then `T::get_mut_pinned` will return a mutable or pinned mutable
    /// reference accordingly.
    ///
    /// # Remarks
    ///
    /// For many types, including [`TempRef`], `T::Mutable` is actually the same as `T::Shared`.
    /// This can be useful when combining mutable and shared references in a tuple. E.g.
    /// `T = (TempRefMut<U>, TempRef<V>)` represents `(&mut U, &V)`, and this is preserved by
    /// [`TempInstPin::new`], whereas [`TempInst::new`] treats it as `(&U, &V)`.
    #[must_use]
    pub fn new(obj: T::Mutable<'a>) -> Self {
        // SAFETY: `obj` is borrowed for `'a`, which outlives the returned instance.
        unsafe { TempInstPin(T::new_temp_mut(obj), PhantomData) }
    }

    /// Dereferences a pinned mutable [`TempInstMut`] reference to obtain a pinned mutable reference
    /// to `T`. From that reference, the original object can be obtained via
    /// [`TempReprMut::get_mut_pinned`].
    #[must_use]
    pub fn deref_pin(self: Pin<&mut Self>) -> Pin<&mut T> {
        unsafe { self.map_unchecked_mut(|inst| &mut inst.0) }
    }

    /// Calls `f` with a pinned mutable reference to `T` which is constructed from `obj`, and
    /// returns the value returned by `f`.
    ///
    /// This method exists for consistency (with respect to [`TempInstMut::call_with`]), but is
    /// actually just a trivial application of [`Self::new`] and [`Self::deref_pin`].
    pub fn call_with<R>(obj: T::Mutable<'a>, f: impl FnOnce(Pin<&mut T>) -> R) -> R {
        let inst = pin!(Self::new(obj));
        f(inst.deref_pin())
    }
}

impl<T: TempReprMut> Deref for TempInstPin<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T: TempReprMut<Mutable<'a>: Default>> Default for TempInstPin<'a, T> {
    fn default() -> Self {
        TempInstPin::new(Default::default())
    }
}

impl<T: TempReprMut + Debug> Debug for TempInstPin<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A wrapper around an instance of `T` which implements [`TempReprMut`], i.e. is a temporary
/// representation of a type `T::Mutable<'a>`.
///
/// [`TempInstMut`] itself has a lifetime parameter `'a` and borrows the object passed to
/// [`TempInstMut::new`] for that lifetime; therefore it is logically equivalent to
/// `T::Mutable<'a>`. However, it can hand out references to the lifetime-less type `T` via its
/// [`DerefMut`] implementation.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TempInstMut<'a, T: TempReprMutChk + 'a>(T, T::SwapChkData, PhantomData<T::Mutable<'a>>);

impl<'a, T: TempReprMutChk> TempInstMut<'a, T> {
    /// Creates a [`TempInstMut`] from an instance of `T::Mutable`.
    ///
    /// A mutable reference to `T` can be obtained from the result via [`DerefMut`].
    ///
    /// As this method is unsafe, using one of the safe alternatives is strongly recommended:
    /// * [`TempInstPin::new`] if a pinned mutable reference to `T` is sufficient.
    /// * [`TempInstMut::call_with`] if the use of the mutable `T` reference can be confined to a
    ///   closure.
    ///
    /// [`TempInstMut::new`] potentially has a slight overhead compared to [`TempInstPin::new`], in
    /// terms of both time and space, though there is a good chance that the compiler will optimize
    /// both away if it can analyze how the instance is used.
    ///
    /// # Safety
    ///
    /// The caller must ensure at least one of the following two conditions.
    /// * The [`Drop`] implementation of [`TempInstMut`] is called whenever the returned instance
    ///   goes out of scope.
    ///   (In particular, the instance must not be passed to [`core::mem::forget`].)
    /// * The state of the instance when it goes out of scope is the same as when it was created.
    ///   (This condition can be violated by calling [`core::mem::swap`] or a related function on
    ///   the [`TempInstMut`] instance. When the instance goes out of scope after passing it to
    ///   [`core::mem::swap`], the _other_ [`TempInstMut`] instance that it was swapped with can
    ///   become dangling. Note that passing the result of [`Self::deref_mut`] to
    ///   [`core::mem::swap`] is not unsafe, however.)
    /// * `'a` is `'static`. (A [`TempInstMut`] instance with static lifetimes cannot become
    ///   dangling.)
    ///
    /// # Panics
    ///
    /// The [`Drop`] implementation of the returned instance calls [`std::process::abort`] (after
    /// calling the standard panic handler) if the instance has been modified, which is not possible
    /// via its API but can be achieved by swapping it with another instance, e.g. using
    /// [`core::mem::swap`].
    ///
    /// Unfortunately, a regular panic is insufficient in this case because it can be caught with
    /// [`std::panic::catch_unwind`], and a dangling [`TempInstMut`] reference can then be obtained
    /// from the closure passed to [`std::panic::catch_unwind`] (safely, because unfortunately
    /// [`std::panic::UnwindSafe`] is not an `unsafe` trait -- why?!!).
    ///
    /// The panic behavior can be changed via [`set_modification_panic_fn`].
    ///
    /// # Remarks
    ///
    /// For many types, including [`TempRef`], `T::Mutable` is actually the same as `T::Shared`.
    /// This can be useful when combining mutable and shared references in a tuple. E.g.
    /// `T = (TempRefMut<U>, TempRef<V>)` represents `(&mut U, &V)`, and this is preserved by
    /// [`TempInstMut::new`], whereas [`TempInst::new`] treats it as `(&U, &V)`.
    #[must_use]
    pub unsafe fn new(obj: T::Mutable<'a>) -> Self {
        // SAFETY: `obj` is borrowed for `'a`, which outlives the returned instance.
        let temp = T::new_temp_mut(obj);
        let chk_data = temp.swap_chk_data();
        TempInstMut(temp, chk_data, PhantomData)
    }

    /// Calls `f` with a mutable reference to `T` constructed from `obj`, and returns the value
    /// returned by `f`.
    ///
    /// This method is a simple (but safe) wrapper around [`Self::new`], which potentially has a
    /// slight overhead. If possible, use [`TempInstPin::new`] (or [`TempInstPin::call_with`])
    /// instead.
    ///
    /// # Panics
    ///
    /// Calls [`std::process::abort`] if `f` modifies the internal state of the object that was
    /// passed to it. See [`Self::new`] for more information.
    pub fn call_with<R>(obj: T::Mutable<'a>, f: impl FnOnce(&mut T) -> R) -> R {
        // SAFETY: Trivial because the scope of `inst` ends at the end of this function.
        let mut inst = unsafe { Self::new(obj) };
        f(&mut inst)
    }
}

impl<T: TempReprMutChk> Deref for TempInstMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: TempReprMutChk> DerefMut for TempInstMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: TempReprMutChk<Mutable<'static>: Default>> Default for TempInstMut<'static, T> {
    fn default() -> Self {
        // SAFETY: Creating a `TempInstMut` instance with a `'static` lifetime is explicitly
        // declared as safe.
        unsafe { TempInstMut::new(Default::default()) }
    }
}

impl<T: TempReprMutChk + Debug> Debug for TempInstMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl<T: TempReprMutChk> Drop for TempInstMut<'_, T> {
    fn drop(&mut self) {
        if self.0.swap_chk_data() != self.1 {
            modification_panic();
        }
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
    panic!("TempInstMut instance was modified; this is not allowed because it violates safety guarantees");
}

#[cfg(not(feature = "std"))]
fn modification_panic_fn() {
    // In the nostd case, we don't have `std::process::abort()`, so entering an endless loop
    // seems to be the best we can do.
    loop {}
}

static mut MODIFICATION_PANIC_FN: fn() = modification_panic_fn;

/// Sets an alternative function to be called by the [`Drop`] implementation of [`TempInstMut`]
/// when it encounters an illegal modification.
///
/// # Safety
///
/// * Changing the panic function is only allowed when no other thread is able to interact with
///   this crate.
///
/// * The panic function must not return unless [`std::thread::panicking`] returns `true`.
///
/// * If the panic function causes an unwinding panic, the caller of
///   [`set_modification_panic_fn`] assumes the responsibility that no instance or reference of
///   [`TempInstMut`] is used across (i.e. captured in) [`std::panic::catch_unwind`].
pub unsafe fn set_modification_panic_fn(panic_fn: fn()) {
    MODIFICATION_PANIC_FN = panic_fn;
}

fn modification_panic() {
    // SAFETY: The safety conditions of `set_modification_panic_fn` guarantee that no other
    // thread is concurrently modifying the static variable.
    unsafe { MODIFICATION_PANIC_FN() }
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
/// * The implementation of the trait must ensure that `new_temp<'a>` followed by `get<'b>` cannot
///   cause undefined behavior when `'a: 'b`. (This essentially restricts `Shared<'a>` to types that
///   are covariant in `'a`.)
///
/// * The above must also hold if a (legal) cast was applied to the result of `new_temp`.
pub unsafe trait TempRepr {
    /// The type that `Self` is a temporary representation of. May contain shared references of
    /// lifetime `'a`.
    type Shared<'a>
    where
        Self: 'a;

    /// Converts the given object to its temporary representation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the returned object does not outlive the lifetime argument of
    /// `obj`, and that [`TempReprMut::get_mut`] or [`TempReprMut::get_mut_pinned`] is never called
    /// on the result.
    ///
    /// (Exception: see the caveat in the safety rules of [`TempReprMutChk`].)
    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self;

    /// Converts from a shared temporary reference back to the original type with a
    /// suitably-restricted lifetime.
    fn get(&self) -> Self::Shared<'_>;
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
/// * `new_temp_mut<'a>` followed by `get_mut<'b>` or `get_mut_pinned<'b>` must not cause undefined
///   behavior when `'a: 'b` and `'b` does not overlap with any other lifetime passed to `get` or
///   `get_mut` or `get_mut_pinned`.
///
/// * `new_temp_mut<'a>` followed by `get<'b>` must not cause undefined behavior when `'a: 'b` and
///   `'b` does not overlap with any lifetime passed to `get_mut` or `get_mut_pinned`.
///
/// * `get_mut_pinned` must either have a custom implementation, or the default implementation via
///   `get_mut` must be safe.
///
/// * The pinning projections that are implemented as part of `get_mut_pinned` in this crate, e.g.
///   for tuples, must be safe when the type is the target of such a projection. (This is usually
///   trivial, and e.g. [`Option::as_pin_mut`] already exists in the standard library, but
///   technically it is not yet automatically the case for tuples.)
///
/// * The above must also hold if a (legal) cast was applied to the result of `new_temp_mut`.
pub unsafe trait TempReprMut: TempRepr {
    /// The type that `Self` is a temporary representation of. May contain mutable references of
    /// lifetime `'a`.
    type Mutable<'a>
    where
        Self: 'a;

    /// Converts the given object to a temporary representation without a lifetime parameter.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the returned object does not outlive the lifetime argument of
    /// `obj`.
    ///
    /// (Exception: see the caveat in the safety rules of [`TempReprMutChk`].)
    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self;

    /// Converts from a mutable reference to the temporary representation back to the original type,
    /// with a restricted lifetime.
    fn get_mut(&mut self) -> Self::Mutable<'_>;

    /// Like [`TempReprMut::get_mut`], but takes a pinned mutable reference to `Self`.
    fn get_mut_pinned(self: Pin<&mut Self>) -> Self::Mutable<'_> {
        // SAFETY: See corresponding trait rule.
        unsafe { self.get_unchecked_mut().get_mut() }
    }
}

/// An extension of [`TempReprMut`] that allows non-pinned mutable references to be passed to safe
/// client code.
///
/// # Safety
///
/// [`TempReprMutChk::swap_chk_data`] must be implemented in such a way that a call to
/// [`core::mem::swap`] is either detectable by calling `swap_chk_data` before and after and
/// comparing the result via the [`PartialEq`] implementation, or is harmless, in the following
/// sense:
///
/// Whenever a swapping operation is not detected by comparing the result of `swap_chk_data`,
/// the swapped instances must be interchangeable in terms of all safety conditions of [`TempRepr`]
/// and [`TempReprMut`].
/// In particular, in all specific points, undefined behavior must also be avoided when the
/// condition "`'a: 'b`" is weakened to "`'c: 'b` for some `'c` where the result of
/// `swap_chk_data` was equal.
pub unsafe trait TempReprMutChk: TempReprMut {
    type SwapChkData: PartialEq;

    /// Obtains an object (usually a copy of the internal state of `self`) that can be used to check
    /// whether `self` was swapped with another instance of `Self` in such a way that the safety
    /// rules of [`TempReprMutChk`] are violated.
    fn swap_chk_data(&self) -> Self::SwapChkData;
}

/// A marker trait that causes [`TempReprMut`] to be implemented identically to [`TempRepr`].
pub trait AlwaysShared: TempRepr {}

// SAFETY: The additional conditions of `TempReprMut` are trivially satisfied if the references
// returned by `get_mut`/`get_mut_pinned` are actually shared references.
unsafe impl<T: AlwaysShared> TempReprMut for T {
    type Mutable<'a> = Self::Shared<'a> where Self: 'a;

    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
        Self::new_temp(obj)
    }

    fn get_mut(&mut self) -> Self::Mutable<'_> {
        self.get()
    }

    fn get_mut_pinned(self: Pin<&mut Self>) -> Self::Mutable<'_> {
        self.into_ref().get_ref().get()
    }
}

/*******************
 * SelfRepr<T> [T] *
 *******************/

/// A wrapper type that trivially implements [`TempRepr`]/[`TempReprMut`] for any `T: Clone` in such
/// a way that no lifetimes are erased.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SelfRepr<T: Clone>(T);

// SAFETY: All conditions are trivially satisfied because the `get` implementation isn't
// actually unsafe.
unsafe impl<T: Clone> TempRepr for SelfRepr<T> {
    type Shared<'a> = T where Self: 'a;

    unsafe fn new_temp(obj: T) -> Self {
        SelfRepr(obj)
    }

    fn get(&self) -> T {
        self.0.clone()
    }
}

impl<T: Clone> AlwaysShared for SelfRepr<T> {}

// SAFETY: Trivially satisfied because the `get` implementation isn't actually unsafe.
unsafe impl<T: Clone> TempReprMutChk for SelfRepr<T> {
    type SwapChkData = ();

    fn swap_chk_data(&self) -> Self::SwapChkData {}
}

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

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        TempRef(obj.into())
    }

    fn get(&self) -> Self::Shared<'_> {
        // SAFETY: The safety rules of `new_temp` and `new_temp_mut` ensure that this call is valid.
        unsafe { self.0.as_ref() }
    }
}

impl<T: ?Sized> AlwaysShared for TempRef<T> {}

// SAFETY: Equal pointers to the same type must point to the same object, unless the object is
// zero-size.
unsafe impl<T: ?Sized> TempReprMutChk for TempRef<T> {
    type SwapChkData = NonNull<T>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        self.0
    }
}

impl<T: ?Sized> Deref for TempRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T: ?Sized + PartialEq> PartialEq for TempRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: ?Sized + Eq> Eq for TempRef<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for TempRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.get().partial_cmp(other.get())
    }

    fn lt(&self, other: &Self) -> bool {
        self.get() < other.get()
    }

    fn le(&self, other: &Self) -> bool {
        self.get() <= other.get()
    }

    fn gt(&self, other: &Self) -> bool {
        self.get() > other.get()
    }

    fn ge(&self, other: &Self) -> bool {
        self.get() >= other.get()
    }
}

impl<T: ?Sized + Ord> Ord for TempRef<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.get().cmp(other.get())
    }
}

impl<T: ?Sized + Hash> Hash for TempRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get().hash(state);
    }
}

impl<T: ?Sized + Debug> Debug for TempRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

// SAFETY: `TempRef<T>` follows the same rules as `&T` regarding thread safety.
unsafe impl<T: ?Sized + Sync> Send for TempRef<T> {}
unsafe impl<T: ?Sized + Sync> Sync for TempRef<T> {}

/**************************
 * TempRefMut<T> [&mut T] *
 **************************/

/// The canonical implementation of [`TempReprMut`], representing a single mutable reference.
///
/// This is not necessarily very useful on its own, but forms the basis of composition via tuples.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TempRefMut<T: ?Sized>(TempRef<T>, PhantomData<*mut T>);

// SAFETY: The safety rules of `TempRepr` are canonically satisfied by conversions between
// shared references and pointers.
unsafe impl<T: ?Sized> TempRepr for TempRefMut<T> {
    type Shared<'a> = &'a T where Self: 'a;

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        TempRefMut(TempRef::new_temp(obj), PhantomData)
    }

    fn get(&self) -> Self::Shared<'_> {
        self.0.get()
    }
}

// SAFETY: The safety rules of `TempReprMut` are canonically satisfied by conversions between
// mutable references and pointers.
//
// The `PhantomData` field guarantees that a `TempRefMut` instance cannot be cast in a covariant
// way, which `NonNull` would allow but violates the safety rules of `TempReprMut`.
unsafe impl<T: ?Sized> TempReprMut for TempRefMut<T> {
    type Mutable<'a> = &'a mut T where Self: 'a;

    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
        TempRefMut(TempRef(obj.into()), PhantomData)
    }

    fn get_mut(&mut self) -> Self::Mutable<'_> {
        // SAFETY: The safety rules of `new_temp` and `new_temp_mut` ensure that this call is valid.
        unsafe { self.0 .0.as_mut() }
    }

    fn get_mut_pinned(mut self: Pin<&mut Self>) -> Self::Mutable<'_> {
        // SAFETY: The safety rules of `new_temp` and `new_temp_mut` ensure that this call is valid.
        unsafe { self.0 .0.as_mut() }
    }
}

// SAFETY: Equal pointers to the same type must point to the same object, unless the object is
// zero-size.
unsafe impl<T: ?Sized> TempReprMutChk for TempRefMut<T> {
    type SwapChkData = NonNull<T>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        self.0.swap_chk_data()
    }
}

impl<T: ?Sized> Deref for TempRefMut<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T: ?Sized> DerefMut for TempRefMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

impl<T: ?Sized + Debug> Debug for TempRefMut<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

// SAFETY: `TempRefMut<T>` follows the same rules as `&mut T` regarding thread safety.
unsafe impl<T: ?Sized + Send> Send for TempRefMut<T> {}
unsafe impl<T: ?Sized + Sync> Sync for TempRefMut<T> {}

/*******************************
 * TempRefPin<T> [Pin<&mut T>] *
 *******************************/

/// Similar to [`TempRefMut`], but represents a pinned mutable reference.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TempRefPin<T: ?Sized>(TempRefMut<T>);

// SAFETY: The safety rules of `TempRepr` are canonically satisfied by conversions between
// shared references and pointers.
unsafe impl<T: ?Sized> TempRepr for TempRefPin<T> {
    type Shared<'a> = &'a T where Self: 'a;

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        TempRefPin(TempRefMut::new_temp(obj))
    }

    fn get(&self) -> Self::Shared<'_> {
        self.0.get()
    }
}

// SAFETY: The safety rules of `TempReprMut` are canonically satisfied by conversions between
// mutable references and pointers, and `Pin<Ptr>` is covariant in `Ptr`.
//
// The `PhantomData` field guarantees that a `TempRefPin` instance cannot be cast in a covariant
// way, which `NonNull` would allow but violates the safety rules of `TempReprMut`.
unsafe impl<T: ?Sized> TempReprMut for TempRefPin<T> {
    type Mutable<'a> = Pin<&'a mut T> where Self: 'a;

    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
        // SAFETY: Converting a pinned reference to a pointer is unproblematic as long as we only
        // convert it back to a pinned or shared reference.
        TempRefPin(TempRefMut::new_temp_mut(obj.get_unchecked_mut()))
    }

    fn get_mut(&mut self) -> Self::Mutable<'_> {
        // SAFETY: The safety rules of `new_temp` and `new_temp_mut` ensure that this call is valid.
        unsafe { Pin::new_unchecked(self.0.get_mut()) }
    }
}

// SAFETY: Equal pointers to the same type must point to the same object, unless the object is
// zero-size.
unsafe impl<T: ?Sized> TempReprMutChk for TempRefPin<T> {
    type SwapChkData = NonNull<T>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        self.0.swap_chk_data()
    }
}

impl<T: ?Sized> Deref for TempRefPin<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T: ?Sized + Debug> Debug for TempRefPin<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

/***********************
 * TempCow<T> [Cow<T>] *
 ***********************/

#[cfg(feature = "alloc")]
pub enum TempCow<T: ?Sized + alloc::borrow::ToOwned> {
    Borrowed(TempRef<T>),
    Owned(T::Owned),
}

#[cfg(feature = "alloc")]
unsafe impl<T: ?Sized + alloc::borrow::ToOwned<Owned: Clone>> TempRepr for TempCow<T> {
    type Shared<'a> = alloc::borrow::Cow<'a, T> where Self: 'a;

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        match obj {
            alloc::borrow::Cow::Borrowed(obj) => TempCow::Borrowed(TempRef::new_temp(obj)),
            alloc::borrow::Cow::Owned(obj) => TempCow::Owned(obj),
        }
    }

    fn get(&self) -> Self::Shared<'_> {
        match self {
            TempCow::Borrowed(temp) => alloc::borrow::Cow::Borrowed(temp.get()),
            TempCow::Owned(temp) => alloc::borrow::Cow::Owned(temp.clone()),
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: ?Sized + alloc::borrow::ToOwned<Owned: Clone>> AlwaysShared for TempCow<T> {}

// SAFETY: Note that we only care about the case where both instances are borrowed. In particular,
// `eq` never returns `true` when one instance is borrowed and the other is owned.
#[cfg(feature = "alloc")]
unsafe impl<T: ?Sized + alloc::borrow::ToOwned<Owned: Clone>> TempReprMutChk for TempCow<T> {
    type SwapChkData = Option<NonNull<T>>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        match self {
            TempCow::Borrowed(temp) => Some(temp.swap_chk_data()),
            TempCow::Owned(_) => None,
        }
    }
}

/*************
 * Option<T> *
 *************/

unsafe impl<T: TempRepr> TempRepr for Option<T> {
    type Shared<'a> = Option<T::Shared<'a>> where Self: 'a;

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        Some(T::new_temp(obj?))
    }

    fn get(&self) -> Self::Shared<'_> {
        Some(self.as_ref()?.get())
    }
}

unsafe impl<T: TempReprMut> TempReprMut for Option<T> {
    type Mutable<'a> = Option<T::Mutable<'a>> where Self: 'a;

    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
        Some(T::new_temp_mut(obj?))
    }

    fn get_mut(&mut self) -> Self::Mutable<'_> {
        Some(self.as_mut()?.get_mut())
    }

    fn get_mut_pinned(self: Pin<&mut Self>) -> Self::Mutable<'_> {
        Some(self.as_pin_mut()?.get_mut_pinned())
    }
}

unsafe impl<T: TempReprMutChk> TempReprMutChk for Option<T> {
    type SwapChkData = Option<T::SwapChkData>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        self.as_ref().map(T::swap_chk_data)
    }
}

/*****************
 * (T0, T1, ...) *
 *****************/

macro_rules! impl_temp_repr_tuple {
    ($($idx:tt $T:ident),*) => {
        #[allow(clippy::unused_unit)]
        unsafe impl<$($T: TempRepr),*> TempRepr for ($($T,)*) {
            type Shared<'a> = ($($T::Shared<'a>,)*) where Self: 'a;

            #[allow(unused_variables)]
            unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
                ($($T::new_temp(obj.$idx),)*)
            }

            fn get(&self) -> Self::Shared<'_> {
                ($(self.$idx.get(),)*)
            }
        }

        #[allow(clippy::unused_unit)]
        unsafe impl<$($T: TempReprMut),*> TempReprMut for ($($T,)*) {
            type Mutable<'a> = ($($T::Mutable<'a>,)*) where Self: 'a;

            #[allow(unused_variables)]
            unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
                ($($T::new_temp_mut(obj.$idx),)*)
            }

            fn get_mut(&mut self) -> Self::Mutable<'_> {
                ($(self.$idx.get_mut(),)*)
            }
        }

        #[allow(clippy::unused_unit)]
        unsafe impl<$($T: TempReprMutChk),*> TempReprMutChk for ($($T,)*) {
            type SwapChkData = ($($T::SwapChkData,)*);

            fn swap_chk_data(&self) -> Self::SwapChkData {
                ($(self.$idx.swap_chk_data(),)*)
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

/******************
 * Either<T0, T1> *
 ******************/

#[cfg(feature = "either")]
unsafe impl<T0: TempRepr, T1: TempRepr> TempRepr for either::Either<T0, T1> {
    type Shared<'a> = either::Either<T0::Shared<'a>, T1::Shared<'a>> where Self: 'a;

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        match obj {
            either::Either::Left(obj0) => either::Either::Left(T0::new_temp(obj0)),
            either::Either::Right(obj1) => either::Either::Right(T1::new_temp(obj1)),
        }
    }

    fn get(&self) -> Self::Shared<'_> {
        match self {
            either::Either::Left(self0) => either::Either::Left(self0.get()),
            either::Either::Right(self1) => either::Either::Right(self1.get()),
        }
    }
}

#[cfg(feature = "either")]
unsafe impl<T0: TempReprMut, T1: TempReprMut> TempReprMut for either::Either<T0, T1> {
    type Mutable<'a> = either::Either<T0::Mutable<'a>, T1::Mutable<'a>> where Self: 'a;

    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
        match obj {
            either::Either::Left(obj0) => either::Either::Left(T0::new_temp_mut(obj0)),
            either::Either::Right(obj1) => either::Either::Right(T1::new_temp_mut(obj1)),
        }
    }

    fn get_mut(&mut self) -> Self::Mutable<'_> {
        match self {
            either::Either::Left(self0) => either::Either::Left(self0.get_mut()),
            either::Either::Right(self1) => either::Either::Right(self1.get_mut()),
        }
    }

    fn get_mut_pinned(self: Pin<&mut Self>) -> Self::Mutable<'_> {
        match self.as_pin_mut() {
            either::Either::Left(self0) => either::Either::Left(self0.get_mut_pinned()),
            either::Either::Right(self1) => either::Either::Right(self1.get_mut_pinned()),
        }
    }
}

#[cfg(feature = "either")]
unsafe impl<T0: TempReprMutChk, T1: TempReprMutChk> TempReprMutChk for either::Either<T0, T1> {
    type SwapChkData = either::Either<T0::SwapChkData, T1::SwapChkData>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        match self {
            either::Either::Left(self0) => either::Either::Left(self0.swap_chk_data()),
            either::Either::Right(self1) => either::Either::Right(self1.swap_chk_data()),
        }
    }
}

/************
 * Range<T> *
 ************/

unsafe impl<T: TempRepr> TempRepr for Range<T> {
    type Shared<'a> = Range<T::Shared<'a>> where Self: 'a;

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        T::new_temp(obj.start)..T::new_temp(obj.end)
    }

    fn get(&self) -> Self::Shared<'_> {
        self.start.get()..self.end.get()
    }
}

unsafe impl<T: TempReprMut> TempReprMut for Range<T> {
    type Mutable<'a> = Range<T::Mutable<'a>> where Self: 'a;

    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
        T::new_temp_mut(obj.start)..T::new_temp_mut(obj.end)
    }

    fn get_mut(&mut self) -> Self::Mutable<'_> {
        self.start.get_mut()..self.end.get_mut()
    }
}

unsafe impl<T: TempReprMutChk> TempReprMutChk for Range<T> {
    type SwapChkData = Range<T::SwapChkData>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        self.start.swap_chk_data()..self.end.swap_chk_data()
    }
}

/****************
 * RangeFrom<T> *
 ****************/

unsafe impl<T: TempRepr> TempRepr for RangeFrom<T> {
    type Shared<'a> = RangeFrom<T::Shared<'a>> where Self: 'a;

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        T::new_temp(obj.start)..
    }

    fn get(&self) -> Self::Shared<'_> {
        self.start.get()..
    }
}

unsafe impl<T: TempReprMut> TempReprMut for RangeFrom<T> {
    type Mutable<'a> = RangeFrom<T::Mutable<'a>> where Self: 'a;

    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
        T::new_temp_mut(obj.start)..
    }

    fn get_mut(&mut self) -> Self::Mutable<'_> {
        self.start.get_mut()..
    }
}

unsafe impl<T: TempReprMutChk> TempReprMutChk for RangeFrom<T> {
    type SwapChkData = RangeFrom<T::SwapChkData>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        self.start.swap_chk_data()..
    }
}

/**************
 * RangeTo<T> *
 **************/

unsafe impl<T: TempRepr> TempRepr for RangeTo<T> {
    type Shared<'a> = RangeTo<T::Shared<'a>> where Self: 'a;

    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
        ..T::new_temp(obj.end)
    }

    fn get(&self) -> Self::Shared<'_> {
        ..self.end.get()
    }
}

unsafe impl<T: TempReprMut> TempReprMut for RangeTo<T> {
    type Mutable<'a> = RangeTo<T::Mutable<'a>> where Self: 'a;

    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
        ..T::new_temp_mut(obj.end)
    }

    fn get_mut(&mut self) -> Self::Mutable<'_> {
        ..self.end.get_mut()
    }
}

unsafe impl<T: TempReprMutChk> TempReprMutChk for RangeTo<T> {
    type SwapChkData = RangeTo<T::SwapChkData>;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        ..self.end.swap_chk_data()
    }
}

/*************
 * RangeFull *
 *************/

unsafe impl TempRepr for RangeFull {
    type Shared<'a> = RangeFull where Self: 'a;

    unsafe fn new_temp(_obj: RangeFull) -> Self {
        ..
    }

    fn get(&self) -> RangeFull {
        ..
    }
}

unsafe impl TempReprMut for RangeFull {
    type Mutable<'a> = RangeFull where Self: 'a;

    unsafe fn new_temp_mut(_obj: RangeFull) -> Self {
        ..
    }

    fn get_mut(&mut self) -> RangeFull {
        ..
    }
}

unsafe impl TempReprMutChk for RangeFull {
    type SwapChkData = RangeFull;

    fn swap_chk_data(&self) -> Self::SwapChkData {
        ..
    }
}

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
        type Temp: TempReprMutChk;

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

        unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
            MappedTempRepr(T::Temp::new_temp(T::shared_to_mapped(obj)))
        }

        fn get(&self) -> Self::Shared<'_> {
            T::mapped_to_shared(self.0.get())
        }
    }

    unsafe impl<T: HasTempRepr> TempReprMut for MappedTempRepr<T> {
        type Mutable<'a> = T::Mutable<'a> where Self: 'a;

        unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
            MappedTempRepr(T::Temp::new_temp_mut(T::mut_to_mapped(obj)))
        }

        fn get_mut(&mut self) -> Self::Mutable<'_> {
            T::mapped_to_mut(self.0.get_mut())
        }

        fn get_mut_pinned(self: Pin<&mut Self>) -> Self::Mutable<'_> {
            unsafe { T::mapped_to_mut(self.map_unchecked_mut(|temp| &mut temp.0).get_mut_pinned()) }
        }
    }

    unsafe impl<T: HasTempRepr> TempReprMutChk for MappedTempRepr<T> {
        type SwapChkData = <T::Temp as TempReprMutChk>::SwapChkData;

        fn swap_chk_data(&self) -> Self::SwapChkData {
            self.0.swap_chk_data()
        }
    }

    impl<T: HasTempRepr> PartialEq for MappedTempRepr<T>
    where
        for<'a> T::Shared<'a>: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            self.get() == other.get()
        }
    }

    impl<T: HasTempRepr> Eq for MappedTempRepr<T> where for<'a> T::Shared<'a>: Eq {}

    impl<T: HasTempRepr> PartialOrd for MappedTempRepr<T>
    where
        for<'a> T::Shared<'a>: PartialOrd,
    {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.get().partial_cmp(&other.get())
        }

        fn lt(&self, other: &Self) -> bool {
            self.get() < other.get()
        }

        fn le(&self, other: &Self) -> bool {
            self.get() <= other.get()
        }

        fn gt(&self, other: &Self) -> bool {
            self.get() > other.get()
        }

        fn ge(&self, other: &Self) -> bool {
            self.get() >= other.get()
        }
    }

    impl<T: HasTempRepr> Ord for MappedTempRepr<T>
    where
        for<'a> T::Shared<'a>: Ord,
    {
        fn cmp(&self, other: &Self) -> Ordering {
            self.get().cmp(&other.get())
        }
    }

    impl<T: HasTempRepr> Hash for MappedTempRepr<T>
    where
        for<'a> T::Shared<'a>: Hash,
    {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.get().hash(state);
        }
    }

    impl<T: HasTempRepr> Debug for MappedTempRepr<T>
    where
        for<'a> T::Shared<'a>: Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.get().fmt(f)
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
                    panic!("TempInstMut instance was modified");
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
        let a_inst = TempInst::<TempRef<i32>>::new(&a);
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
        let double = TempInstMut::<TempRefMut<i32>>::call_with(&mut a, |a_inst| {
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
        let sum = TempInstMut::<(TempRefMut<i32>, TempRefMut<i32>)>::call_with(
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
            TempInstMut::<(TempRefMut<i32>, TempRef<i32>)>::call_with((&mut a, &b), |a_b_inst| {
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
        let a_inst = pin!(TempInstPin::<TempRefMut<i32>>::new(&mut a));
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
        let double = TempInstPin::<TempRefMut<i32>>::call_with(&mut a, |a_inst| {
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
        let double = TempInstPin::<TempRefPin<i32>>::call_with(Pin::as_mut(&mut a), |a_inst| {
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
    #[should_panic(expected = "TempInstMut instance was modified")]
    fn temp_ref_call_mut_illegal_swap() {
        init_tests();
        let mut a = 42;
        TempInstMut::<TempRefMut<i32>>::call_with(&mut a, |a_inst| {
            let mut b = 43;
            TempInstMut::<TempRefMut<i32>>::call_with(&mut b, |b_inst| {
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
        TempInstMut::<TempRefMut<()>>::call_with(&mut a.0, |a0_inst| {
            TempInstMut::<TempRefMut<()>>::call_with(&mut a.1, |a1_inst| {
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
        let a_inst = TempInst::<TempRef<i32>>::new(&a);
        std::thread::scope(|scope| {
            let thread = scope.spawn(move || **a_inst);
            let result = thread.join().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_sync() {
        init_tests();
        let a = 42;
        let a_inst = TempInst::<TempRef<i32>>::new(&a);
        std::thread::scope(|scope| {
            let thread = scope.spawn(|| **a_inst);
            let result = thread.join().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_pin_send() {
        init_tests();
        let a = pin!(42);
        let a_inst = TempInstPin::<TempRefPin<i32>>::new(a);
        std::thread::scope(|scope| {
            let thread = scope.spawn(move || **a_inst);
            let result = thread.join().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_pin_sync() {
        init_tests();
        let a = pin!(42);
        let a_inst = TempInstPin::<TempRefPin<i32>>::new(a);
        std::thread::scope(|scope| {
            let thread = scope.spawn(|| **a_inst);
            let result = thread.join().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn temp_ref_call_mut_send() {
        init_tests();
        let mut a = 42;
        TempInstMut::<TempRefMut<i32>>::call_with(&mut a, |a_inst| {
            std::thread::scope(|scope| {
                let thread = scope.spawn(move || **a_inst += 1);
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
        TempInstMut::<TempRefMut<i32>>::call_with(&mut a, |a_inst| {
            std::thread::scope(|scope| {
                let thread = scope.spawn(|| **a_inst += 1);
                thread.join().unwrap();
            })
        });
        assert_eq!(a, 43);
    }
}
