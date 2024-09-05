# temp-inst crate

[`TempInst`](temp-inst/src/lib.rs) is a safe lifetime-erased wrapper for a Rust object (struct,
enum, tuple, ...) with lifetime parameters. It is intended for use with APIs that require a single
(shared or mutable) reference to an object that cannot depend on lifetimes (except for the lifetime
of that single reference).

For example, `TempInst` can be used to contract a tuple of references into a single reference
and then later back into a tuple of references. For the full list of types that can be used
with `TempInst`, see the implementations of the `TempRepr` trait.

## Example

```rust
// We want to use this external function, which takes a single mutable reference to some `T`
// (with a useless `'static` bound because otherwise there will be a trivial solution).
fn run_twice<T: 'static>(obj: &mut T, f: fn(&mut T)) {
    f(obj);
    f(obj);
}

// However, here we have two separate variables, and we want to pass references to both of them
// to `run_twice`. `T = (&mut a, &mut b)` won't work because of the `'static` bound. (In a
// real-world use case, the problem is usually that `T` cannot have lifetime parameters for some
// other reason.)
let mut a = 42;
let mut b = 23;

// The lambda we pass to `call_with_mut` receives a single mutable reference
// `inst: &mut TempInst<_>` that we can use.
TempInst::<(TempRefMut<i32>, TempRefMut<i32>)>::call_with_mut((&mut a, &mut b), |inst| {
    run_twice(inst, |inst| {
        // Now that we have passed `inst` through the external `run_twice` function back to our
        // own code, we can extract the original pair of references from it.
        let (a_ref, b_ref) = inst.get_mut();
        *a_ref += *b_ref;
        *b_ref += 1;
    })
});

assert_eq!(a, 42 + 23 + 1 + 23);
assert_eq!(b, 23 + 1 + 1);
```

See the [source code](temp-inst/src/lib.rs) for more info.

## Similar crates

[dyn-context](https://crates.io/crates/dyn-context) is another solution for the same problem. The
main difference is that it focuses on user-defined types, whereas temp-inst defines a trait that is
already implemented for standard types but must currently be implemented manually for custom types.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
