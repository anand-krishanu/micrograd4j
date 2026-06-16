package io.github.anandkrishanu.micrograd;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ValueTest {

    private static final double EPS = 1e-4;

    @Test
    void addForwardAndBackward() {
        Value a = new Value(2.0);
        Value b = new Value(-3.0);
        Value c = a.add(b);
        c.backward();

        assertEquals(-1.0, c.data, EPS);
        assertEquals(1.0, a.grad, EPS);   // d(a+b)/da = 1
        assertEquals(1.0, b.grad, EPS);   // d(a+b)/db = 1
    }

    @Test
    void multiplyBackwardUsesOtherOperand() {
        Value a = new Value(2.0);
        Value b = new Value(-3.0);
        Value c = a.multiply(b);
        c.backward();

        assertEquals(-6.0, c.data, EPS);
        assertEquals(-3.0, a.grad, EPS);  // d(a*b)/da = b
        assertEquals(2.0, b.grad, EPS);   // d(a*b)/db = a
    }

    @Test
    void reusedNodeAccumulatesGradient() {
        Value x = new Value(3.0);
        Value y = x.add(x);   // 2x  -> dy/dx should be 2, not 1
        y.backward();

        assertEquals(6.0, y.data, EPS);
        assertEquals(2.0, x.grad, EPS);
    }

    @Test
    void powerBackward() {
        Value x = new Value(3.0);
        Value y = x.power(2);   // x^2 -> dy/dx = 2x = 6
        y.backward();

        assertEquals(9.0, y.data, EPS);
        assertEquals(6.0, x.grad, EPS);
    }

    @Test
    void reluGatesNegativeInputs() {
        Value pos = new Value(2.0).relu();
        pos.backward();
        assertEquals(2.0, pos.data, EPS);

        Value neg = new Value(-2.0);
        Value out = neg.relu();
        out.backward();
        assertEquals(0.0, out.data, EPS);
        assertEquals(0.0, neg.grad, EPS);   // gradient blocked for x < 0
    }

    @Test
    void zeroGradResetsWholeGraph() {
        Value a = new Value(2.0);
        Value b = a.multiply(3.0);
        b.backward();
        assertTrue(a.grad != 0.0);

        b.zeroGrad();
        assertEquals(0.0, a.grad, EPS);
        assertEquals(0.0, b.grad, EPS);
    }

    /**
     * The canonical micrograd example. Reference gradients come from PyTorch.
     */
    @Test
    void karpathyReferenceExample() {
        Value a = new Value(-4.0);
        Value b = new Value(2.0);

        Value c = a.add(b);
        Value d = a.multiply(b).add(b.power(3));
        c = c.add(c.add(1));
        c = c.add(new Value(1).add(c).add(a.negate()));
        d = d.add(d.multiply(2).add(b.add(a).relu()));
        d = d.add(d.multiply(3).add(b.subtract(a).relu()));
        Value e = c.subtract(d);
        Value f = e.power(2);
        Value g = f.divide(2.0);
        g = g.add(new Value(10.0).divide(f));

        g.backward();

        assertEquals(24.7041, g.data, EPS);
        assertEquals(138.8338, a.grad, EPS);
        assertEquals(645.5773, b.grad, EPS);
    }
}
