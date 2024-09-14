//! [`TempStack`] is a linked list data structure based on the [`temp_inst`] crate. The intended use
//! case is that list items are allocated on the call stack; then the list also represents a "stack"
//! with "frames". Via [`temp_inst`], each frame can contain references to data that is available at
//! the point where it is constructed, without having to add lifetime parameters.
//!
//! # Example
//!
//! The following lambda expression parser uses [`TempStack`] as a context that specifies which
//! variables are in scope, in order to determine the
//! [de Bruijn index](https://en.wikipedia.org/wiki/De_Bruijn_index) corresponding to a given
//! variable name.
//!
//! ```
//! # use temp_inst::TempRef;
//! # use crate::temp_stack::TempStack;
//! #
//! #[derive(Clone, PartialEq, Debug)]
//! enum Expr {
//!     Var(usize), // A de Bruijn index that specifies which binder the variable references.
//!     App(Box<Expr>, Box<Expr>),
//!     Lambda(String, Box<Expr>),
//! }
//!
//! // The context containing the variables that are in scope at any given point during
//! // parsing. Note how `Ctx` does not require any lifetime parameters, even though it
//! // references strings with arbitrary lifetimes.
//! type Ctx = TempStack<(), TempRef<str>>;
//!
//! fn parse(s: &str) -> Result<Expr, String> {
//!     let root_ctx = Ctx::new_root(());
//!     let (expr, s) = parse_expr(s, &root_ctx)?;
//!     if !s.is_empty() {
//!         return Err(format!("unexpected character at `{s}`"));
//!     }
//!     Ok(expr)
//! }
//!
//! fn parse_expr<'a>(s: &'a str, ctx: &Ctx) -> Result<(Expr, &'a str), String> {
//!     let (expr, mut s) = parse_single_expr(s, ctx)?;
//!     let Some(mut expr) = expr else {
//!         return Err(format!("expected expression at `{s}`"));
//!     };
//!     loop {
//!         let (arg, r) = parse_single_expr(s, ctx)?;
//!         s = r;
//!         let Some(arg) = arg else {
//!             break;
//!         };
//!         expr = Expr::App(Box::new(expr), Box::new(arg));
//!     }
//!     Ok((expr, s))
//! }
//!
//! fn parse_single_expr<'a>(s: &'a str, ctx: &Ctx) -> Result<(Option<Expr>, &'a str), String> {
//!     let s = s.trim_ascii_start();
//!     if let Some(s) = s.strip_prefix('λ') {
//!         let s = s.trim_ascii_start();
//!         let name_len = s
//!             .find(|ch: char| !ch.is_ascii_alphanumeric())
//!             .unwrap_or(s.len());
//!         if name_len == 0 {
//!             return Err(format!("expected parameter name at `{s}`"));
//!         }
//!         let (name, s) = s.split_at(name_len);
//!         let s = s.trim_ascii_start();
//!         let Some(s) = s.strip_prefix('.') else {
//!             return Err(format!("expected `.` at `{s}`"));
//!         };
//!         // Create a new context with `name` added.
//!         let body_ctx = ctx.new_frame(name);
//!         let (body, s) = parse_expr(s, &body_ctx)?;
//!         Ok((Some(Expr::Lambda(name.into(), Box::new(body))), s))
//!     } else if let Some(s) = s.strip_prefix('(') {
//!         let (body, s) = parse_expr(s, ctx)?;
//!         let Some(s) = s.strip_prefix(')') else {
//!             return Err(format!("expected `)` at `{s}`"));
//!         };
//!         Ok((Some(body), s))
//!     } else {
//!         let name_len = s
//!             .find(|ch: char| !ch.is_ascii_alphanumeric())
//!             .unwrap_or(s.len());
//!         if name_len == 0 {
//!             Ok((None, s))
//!         } else {
//!             let (name, r) = s.split_at(name_len);
//!             // Determine the De Bruijn index of the nearest `name` in context.
//!             let Some(idx) = ctx.iter().position(|v| v == name) else {
//!                 return Err(format!("variable `{name}` not found at `{s}`"));
//!             };
//!             Ok((Some(Expr::Var(idx)), r))
//!         }
//!     }
//! }
//!
//! assert_eq!(
//!     parse("λx.x"),
//!     Ok(Expr::Lambda("x".into(), Box::new(Expr::Var(0))))
//! );
//!
//! assert_eq!(
//!     parse("λx. x x"),
//!     Ok(Expr::Lambda(
//!         "x".into(),
//!         Box::new(Expr::App(Box::new(Expr::Var(0)), Box::new(Expr::Var(0))))
//!     ))
//! );
//!
//! assert_eq!(
//!     parse("λx.λy. y (x y x)"),
//!     Ok(Expr::Lambda(
//!         "x".into(),
//!         Box::new(Expr::Lambda(
//!             "y".into(),
//!             Box::new(Expr::App(
//!                 Box::new(Expr::Var(0)),
//!                 Box::new(Expr::App(
//!                     Box::new(Expr::App(Box::new(Expr::Var(1)), Box::new(Expr::Var(0)))),
//!                     Box::new(Expr::Var(1)),
//!                 ))
//!             ))
//!         ))
//!     ))
//! );
//!
//! assert_eq!(
//!     parse("λx.λy. (λz.z) (x z x)"),
//!     Err("variable `z` not found at `z x)`".into())
//! );
//! ```

#![no_std]

use core::{fmt::Debug, iter::FusedIterator, mem::take, pin::Pin};

use either::Either;
use temp_inst::{TempInst, TempInstPin, TempRefPin, TempRepr, TempReprMut};

/// A linked list consisting of a single item of type `Root` and arbitrarily many items of type
/// `Frame`. Both types must implement [`temp_inst::TempRepr`], which declares them as "temporary
/// representations" of possibly lifetime-dependent types such as references.
///
/// A [`TempStack`] can be constructed and referenced in a mutable or shared fashion, and in the
/// mutable case the usual exclusivity rules apply. However, adding an item never alters the list
/// it was added to; it merely creates a new list that borrows the original one (exclusively or
/// shared).
///
/// # Remarks
///
/// Although the root and frames can consist of arbitrary data via [`temp_inst::SelfRepr`], usually
/// the size of both should be kept small, using references via [`temp_inst::TempRef`] or
/// [`temp_inst::TempRefMut`] instead, for two reasons.
/// * Both root and frame data are stored in the same `enum`, so a large root also enlarges each
///   frame.
/// * The iterators return copies/clones of the frame data. Therefore, if frames are large,
///   iteration should be implemented manually.
#[derive(TempRepr, TempReprMut)]
pub enum TempStack<Root: TempRepr, Frame: TempRepr> {
    Root {
        data: Root,
    },
    Frame {
        data: Frame,
        parent: TempRefPin<TempStack<Root, Frame>>,
    },
}

impl<Root: TempRepr, Frame: TempRepr> TempStack<Root, Frame> {
    /// Creates a new stack and returns a [`TempInst`] object that only hands out shared references.
    pub fn new_root(data: Root::Shared<'_>) -> TempStackFrame<'_, Root, Frame> {
        TempInst::new(Either::Left(data))
    }

    /// Creates a new stack that extends `self` with the given frame, and returns a [`TempInst`]
    /// object that only hands out shared references.
    pub fn new_frame<'a>(&'a self, data: Frame::Shared<'a>) -> TempStackFrame<'a, Root, Frame> {
        TempInst::new(Either::Right((data, self)))
    }

    /// Returns an iterator that traverses the stack starting at the current frame and ending at the
    /// root.
    ///
    /// The iterator returns the data of each frame, and also provides [`TempStackIter::into_root`]
    /// to access the root data.
    pub fn iter(&self) -> TempStackIter<'_, Root, Frame> {
        TempStackIter::new(self)
    }
}

impl<Root: TempReprMut, Frame: TempReprMut> TempStack<Root, Frame> {
    /// Creates a new stack and returns a [`TempInstPin`] object that can hand out pinned mutable
    /// references.
    ///
    /// Note that this requires the resulting object to be pinned, e.g. using [`core::pin::pin!`].
    pub fn new_root_mut(data: Root::Mutable<'_>) -> TempStackFrameMut<'_, Root, Frame> {
        TempInstPin::new(Either::Left(data))
    }

    /// Creates a new stack that extends `self` with the given frame, and returns a [`TempInstPin`]
    /// object that can hand out pinned mutable references.
    ///
    /// Note that this requires the resulting object to be pinned, e.g. using [`core::pin::pin!`].
    pub fn new_frame_mut<'a>(
        self: Pin<&'a mut Self>,
        data: Frame::Mutable<'a>,
    ) -> TempStackFrameMut<'a, Root, Frame> {
        TempInstPin::new(Either::Right((data, self)))
    }

    /// Returns an iterator that traverses the stack starting at the current frame and ending at the
    /// root, returning mutable frames.
    ///
    /// The iterator returns the data of each frame, and also provides
    /// [`TempStackIterMut::into_root`] to access the root data.
    pub fn iter_mut(self: Pin<&mut Self>) -> TempStackIterMut<'_, Root, Frame> {
        TempStackIterMut::new(self)
    }
}

impl<Root: TempRepr + Debug, Frame: TempRepr + Debug> Debug for TempStack<Root, Frame> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("[")?;
        self.fmt_contents(f)?;
        f.write_str("]")?;
        Ok(())
    }
}

impl<Root: TempRepr + Debug, Frame: TempRepr + Debug> TempStack<Root, Frame> {
    fn fmt_contents(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TempStack::Root { data } => data.fmt(f),
            TempStack::Frame { data, parent } => {
                parent.fmt_contents(f)?;
                let separator = if matches!(**parent, TempStack::Root { .. }) {
                    "; "
                } else {
                    ", "
                };
                f.write_str(separator)?;
                data.fmt(f)
            }
        }
    }
}

pub type TempStackRef<'a, Root, Frame> = &'a TempStack<Root, Frame>;
pub type TempStackRefMut<'a, Root, Frame> = Pin<&'a mut TempStack<Root, Frame>>;

pub type TempStackFrame<'a, Root, Frame> = TempInst<'a, TempStack<Root, Frame>>;
pub type TempStackFrameMut<'a, Root, Frame> = TempInstPin<'a, TempStack<Root, Frame>>;

/// An iterator over frames of a shared `TempStack`.
pub struct TempStackIter<'a, Root: TempRepr, Frame: TempRepr>(TempStackRef<'a, Root, Frame>);

impl<'a, Root: TempRepr, Frame: TempRepr> TempStackIter<'a, Root, Frame> {
    fn new(start: TempStackRef<'a, Root, Frame>) -> Self {
        TempStackIter(start)
    }

    /// Consumes the iterator and returns the root data of the stack.
    /// This method is cheap if the iterator has already reached the end, but needs to traverse the
    /// rest of the stack if it has not.
    pub fn into_root(mut self) -> Root::Shared<'a> {
        loop {
            match self.0 {
                TempStack::Root { data } => {
                    return data.get();
                }
                TempStack::Frame { parent, .. } => {
                    self.0 = parent.get();
                }
            }
        }
    }
}

impl<'a, Root: TempRepr, Frame: TempRepr> Copy for TempStackIter<'a, Root, Frame> {}

impl<'a, Root: TempRepr, Frame: TempRepr> Clone for TempStackIter<'a, Root, Frame> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Root: TempRepr, Frame: TempRepr> Iterator for TempStackIter<'a, Root, Frame> {
    type Item = Frame::Shared<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0 {
            TempStack::Root { .. } => None,
            TempStack::Frame { data, parent } => {
                self.0 = parent.get();
                Some(data.get())
            }
        }
    }
}

impl<'a, Root: TempRepr, Frame: TempRepr> FusedIterator for TempStackIter<'a, Root, Frame> {}

/// An iterator over frames of a mutable `TempStack`.
pub struct TempStackIterMut<'a, Root: TempReprMut, Frame: TempReprMut>(
    // Note that this should never be `None`, but we temporarily need to extract the value in the
    // `next` method.
    Option<TempStackRefMut<'a, Root, Frame>>,
);

impl<'a, Root: TempReprMut, Frame: TempReprMut> TempStackIterMut<'a, Root, Frame> {
    fn new(start: TempStackRefMut<'a, Root, Frame>) -> Self {
        TempStackIterMut(Some(start))
    }

    /// Consumes the iterator and returns the root data of the stack.
    /// This method is cheap if the iterator has already reached the end, but needs to traverse the
    /// rest of the stack if it has not.
    pub fn into_root(self) -> Root::Mutable<'a> {
        let mut temp = self.0.unwrap();
        // SAFETY: This only implements a pinning projection.
        unsafe {
            loop {
                match temp.get_unchecked_mut() {
                    TempStack::Root { data } => {
                        return Pin::new_unchecked(data).get_mut_pinned();
                    }
                    TempStack::Frame { parent, .. } => {
                        temp = Pin::new_unchecked(parent).get_mut_pinned();
                    }
                }
            }
        }
    }
}

impl<'a, Root: TempReprMut, Frame: TempReprMut> Iterator for TempStackIterMut<'a, Root, Frame> {
    type Item = Frame::Mutable<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let temp = take(&mut self.0).unwrap();
        // SAFETY: This only implements a pinning projection.
        unsafe {
            let temp = temp.get_unchecked_mut();
            match temp {
                TempStack::Root { .. } => {
                    self.0 = Some(Pin::new_unchecked(temp));
                    None
                }
                TempStack::Frame { data, parent } => {
                    self.0 = Some(Pin::new_unchecked(parent).get_mut_pinned());
                    Some(Pin::new_unchecked(data).get_mut_pinned())
                }
            }
        }
    }
}

impl<'a, Root: TempReprMut, Frame: TempReprMut> FusedIterator
    for TempStackIterMut<'a, Root, Frame>
{
}

#[cfg(test)]
mod tests {
    use core::pin::pin;

    use temp_inst::{TempRef, TempRefMut};

    use super::*;

    #[test]
    fn empty_stack() {
        let root = 42;
        let stack = TempStack::<TempRef<i32>, ()>::new_root(&root);

        let mut iter = stack.iter();
        assert!(iter.next().is_none());
        let root_ref = iter.into_root();
        assert_eq!(*root_ref, 42);
    }

    #[test]
    fn empty_stack_mut() {
        let mut root = 42;
        let stack = pin!(TempStack::<TempRefMut<i32>, ()>::new_root_mut(&mut root));

        let mut iter = stack.deref_pin().iter_mut();
        assert!(iter.next().is_none());
        let root_ref = iter.into_root();
        assert_eq!(*root_ref, 42);
        *root_ref += 1;
        assert_eq!(root, 43);
    }

    #[test]
    fn stack_with_frames() {
        let root = 42;
        let stack = TempStack::<TempRef<i32>, TempRef<i32>>::new_root(&root);
        let stack = stack.new_frame(&1);
        let stack = stack.new_frame(&2);
        let stack = stack.new_frame(&3);

        let mut iter = stack.iter();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert!(iter.next().is_none());
        let root_ref = iter.into_root();
        assert_eq!(*root_ref, 42);

        let iter = stack.iter();
        let root_ref = iter.into_root();
        assert_eq!(*root_ref, 42);
    }

    #[test]
    fn stack_with_frames_mut() {
        let mut root = 42;
        let stack = pin!(TempStack::<TempRefMut<i32>, TempRefMut<i32>>::new_root_mut(
            &mut root
        ));
        let mut frame1 = 1;
        let stack = pin!(stack.deref_pin().new_frame_mut(&mut frame1));
        let mut frame2 = 2;
        let stack = pin!(stack.deref_pin().new_frame_mut(&mut frame2));
        let mut frame3 = 3;
        let mut stack = pin!(stack.deref_pin().new_frame_mut(&mut frame3));

        let mut iter = stack.as_mut().deref_pin().iter_mut();
        let frame3_ref = iter.next().unwrap();
        assert_eq!(frame3_ref, &mut 3);
        *frame3_ref += 1;
        assert_eq!(iter.next(), Some(&mut 2));
        let frame1_ref = iter.next().unwrap();
        assert_eq!(frame1_ref, &mut 1);
        *frame1_ref -= 1;
        assert!(iter.next().is_none());
        let root_ref = iter.into_root();
        assert_eq!(*root_ref, 42);
        *root_ref += 1;

        let iter = stack.deref_pin().iter_mut();
        let root_ref = iter.into_root();
        assert_eq!(*root_ref, 43);

        assert_eq!(root, 43);
        assert_eq!(frame1, 0);
        assert_eq!(frame2, 2);
        assert_eq!(frame3, 4);
    }

    #[test]
    fn stack_with_branching() {
        let root = 42;
        let stack = TempStack::<TempRef<i32>, TempRef<i32>>::new_root(&root);
        let stack = stack.new_frame(&1);
        let stack = stack.new_frame(&2);
        let stack2 = stack.new_frame(&11);
        let stack = stack.new_frame(&3);
        let stack2 = stack2.new_frame(&12);
        let stack2 = stack2.new_frame(&13);

        let mut iter = stack.iter();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert!(iter.next().is_none());

        let mut iter2 = stack2.iter();
        assert_eq!(iter2.next(), Some(&13));
        assert_eq!(iter2.next(), Some(&12));
        assert_eq!(iter2.next(), Some(&11));
        assert_eq!(iter2.next(), Some(&2));
        assert_eq!(iter2.next(), Some(&1));
        assert!(iter2.next().is_none());
    }

    #[test]
    fn stack_with_branching_mut() {
        let mut root = 42;
        let stack = pin!(TempStack::<TempRefMut<i32>, TempRefMut<i32>>::new_root_mut(
            &mut root
        ));
        let mut frame1 = 1;
        let stack = pin!(stack.deref_pin().new_frame_mut(&mut frame1));
        let mut frame2 = 2;
        let mut stack = pin!(stack.deref_pin().new_frame_mut(&mut frame2));
        let mut frame3 = 3;
        let stack2 = pin!(stack.as_mut().deref_pin().new_frame_mut(&mut frame3));

        let mut iter = stack2.deref_pin().iter_mut();
        let frame3_ref = iter.next().unwrap();
        assert_eq!(frame3_ref, &mut 3);
        *frame3_ref += 1;
        assert_eq!(iter.next(), Some(&mut 2));
        let frame1_ref = iter.next().unwrap();
        assert_eq!(frame1_ref, &mut 1);
        *frame1_ref -= 1;
        assert!(iter.next().is_none());
        let root_ref = iter.into_root();
        assert_eq!(*root_ref, 42);
        *root_ref += 1;

        let mut iter = stack.deref_pin().iter_mut();
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), Some(&mut 0));
        assert!(iter.next().is_none());
        let root_ref = iter.into_root();
        assert_eq!(*root_ref, 43);

        assert_eq!(root, 43);
        assert_eq!(frame1, 0);
        assert_eq!(frame2, 2);
        assert_eq!(frame3, 4);
    }
}
