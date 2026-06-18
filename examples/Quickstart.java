///usr/bin/env jbang "$0" "$@" ; exit $?
//REPOS jitpack=https://jitpack.io
//DEPS com.github.anand-krishanu:micrograd4j:v1.1.0
//
// Zero-install demo. With JBang installed (https://www.jbang.dev), run:
//     jbang examples/Quickstart.java
// JBang fetches micrograd4j from JitPack and runs this file -- no clone, no build.
// (Requires a published tag 'v1.1.0' on the repo so JitPack can build it.)

import io.github.anandkrishanu.micrograd.Value;

public class Quickstart {
    public static void main(String[] args) {
        // Build an expression, then differentiate it.
        Value a = new Value(2.0);
        Value b = new Value(-3.0);
        Value c = new Value(10.0);
        Value e = a.multiply(b);     // -6
        Value d = e.add(c);          //  4
        Value f = new Value(-2.0);
        Value L = d.multiply(f);     // -8

        L.backward();

        System.out.println("L = " + L.data);     // -8.0
        System.out.println("dL/da = " + a.grad);  // 6.0
        System.out.println("dL/db = " + b.grad);  // -4.0
    }
}
