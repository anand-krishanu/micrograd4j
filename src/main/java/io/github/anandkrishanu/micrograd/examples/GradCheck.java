package io.github.anandkrishanu.micrograd.examples;

import io.github.anandkrishanu.micrograd.Value;

/**
 * Sanity check for Value.backward() against the canonical micrograd example.
 * Reference values come from PyTorch: g = 24.7041, dg/da = 138.8338, dg/db = 645.5773.
 *
 * Run: java io.github.anandkrishanu.micrograd.examples.GradCheck
 */
public class GradCheck {
    public static void main(String[] args) {
        Value a = new Value(-4.0);
        Value b = new Value(2.0);

        Value c = a.add(b);
        Value d = a.multiply(b).add(b.power(3));
        c = c.add(c.add(1));                                 // c += c + 1
        c = c.add(new Value(1).add(c).add(a.negate()));      // c += 1 + c + (-a)
        d = d.add(d.multiply(2).add(b.add(a).relu()));       // d += d * 2 + (b + a).relu()
        d = d.add(d.multiply(3).add(b.subtract(a).relu()));  // d += 3 * d + (b - a).relu()
        Value e = c.subtract(d);
        Value f = e.power(2);
        Value g = f.divide(2.0);                             // g = f / 2.0
        g = g.add(new Value(10.0).divide(f));                // g += 10.0 / f

        g.backward();

        System.out.printf("g     = %.4f   (expected 24.7041)%n", g.data);
        System.out.printf("dg/da = %.4f  (expected 138.8338)%n", a.grad);
        System.out.printf("dg/db = %.4f  (expected 645.5773)%n", b.grad);
    }
}
