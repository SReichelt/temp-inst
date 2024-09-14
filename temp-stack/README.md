# temp-stack crate

`TempStack` is a linked list data structure based on the [temp-inst](https://crates.io/crates/temp-inst)
crate. The intended use case is that list items are allocated on the call stack; then the list also
represents a "stack" with "frames". Via temp-inst, each frame can contain references to data that is
available at the point where it is constructed, without having to add lifetime parameters.

## Example

A parser or compiler or interpreter can use a `TempStack` reference as a context that is passed to
individual functions, to determine which variables are in scope. For example, a context with just
variable names might be defined as

```rust
type Ctx = TempStack<(), TempRef<str>>;
```

Due to the use of `TempRef` from the [temp-inst](https://crates.io/crates/temp-inst) crate, no
lifetime parameters are required to access local string slices.

A function can construct construct a new context with an added variable, and pass it to another
function:

```rust
fn parse_expr<'a>(s: &'a str, ctx: &Ctx) -> (Expr, &'a str) {
    // ...
    if is_lambda {
        let (s, name) = parse_name(s);
        let body_ctx = ctx.new_frame(name);
        return parse_expr(s, &body_ctx);
    }
    // ...
}
```

We can easily iterate over a context to find a given variable:

```rust
fn parse_expr<'a>(s: &'a str, ctx: &Ctx) -> (Expr, &'a str) {
    // ...
    if is_var {
        let (s, name) = parse_name(s);
        // Determine the De Bruijn index of the nearest `name` in context.
        let Some(idx) = ctx.iter().position(|v| v == name) else ...
        // ...
    }
}
```

See the [documentation](https://docs.rs/temp-stack/latest/temp_stack/) for the complete example.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT License
   ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
