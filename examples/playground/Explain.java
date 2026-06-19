import io.github.anandkrishanu.micrograd.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Turns the "how does this work?" question into words: a plain-English local-derivative line for
 * any node in the graph, plus a short guided intro to reverse-mode autodiff. Reads only public
 * fields of {@link Value}.
 */
final class Explain {
    private Explain() {}

    /** A one-line explanation of the local backward rule applied at {@code v}, with live numbers. */
    static String localRule(Value v) {
        String op = v.op == null ? "" : v.op;
        if (op.isEmpty() || v.prev.isEmpty()) {
            return "leaf — no local rule; it just accumulates gradient from every path it feeds.";
        }
        double g = v.grad;
        if (op.startsWith("^")) {
            String k = op.substring(1);
            return String.format("power(^%s): ∂/∂x = %s·x^(%s−1); grad ×= that  (upstream g=%+.3f)", k, k, k, g);
        }
        switch (op) {
            case "+":
                return String.format("add: ∂(x+y)/∂x = 1, so each input's grad += upstream g=%+.3f, unchanged.", g);
            case "-":
                return String.format("subtract: first input gets +g, second gets −g  (g=%+.3f)", g);
            case "*": {
                String vals = new ArrayList<>(v.prev).stream().map(c -> fmt(c.data)).collect(Collectors.joining(" , "));
                return String.format("multiply: each input's grad += (product of the others)×g. inputs=[%s], g=%+.3f", vals, g);
            }
            case "/":
                return String.format("divide: quotient rule — numerator gets g/denom, denom gets −g·num/denom²  (g=%+.3f)", g);
            case "tanh":
                return String.format("tanh: ∂tanh/∂x = 1 − tanh² = 1 − %.3f² = %.3f; grad ×= that  (g=%+.3f)",
                        v.data, 1 - v.data * v.data, g);
            case "relu": case "ReLU": case "RELU":
                return String.format("relu: gradient passes only where input>0; output=%.3f so it %s  (g=%+.3f)",
                        v.data, v.data > 0 ? "passes" : "is blocked", g);
            case "exp":
                return String.format("exp: ∂eˣ/∂x = eˣ = output = %.3f; grad ×= that  (g=%+.3f)", v.data, g);
            default:
                return String.format("%s: applies the chain rule, passing upstream g=%+.3f to its inputs.", op, g);
        }
    }

    private static String fmt(double d) {
        return String.format("%.3f", d);
    }

    /** The pages of the "How autograd works" guided tour (panel bodies). */
    static List<String> intro() {
        List<String> pages = new ArrayList<>();
        pages.add(String.join("\n",
                "Autograd in three sentences:",
                "",
                "1. Every number you compute is a Value that remembers how it was made:",
                "   its data, the inputs it came from (prev), and the operation (op).",
                "2. Those links form a graph — a DAG — from your inputs up to the result.",
                "3. backward() walks that graph in reverse, using the chain rule to fill in",
                "   each Value's grad = how much the result changes if you nudge that node."
        ));
        pages.add(String.join("\n",
                "The chain rule, intuitively:",
                "",
                "  dL/dx = (how L reacts to this op) × (how this op reacts to x)",
                "",
                "We seed the output's grad = 1.0, then hand gradient backward one node at a",
                "time. Each op only needs to know its own LOCAL derivative; multiplying those",
                "along every path (and summing where paths meet) gives every input's gradient."
        ));
        pages.add(String.join("\n",
                "Local rules each op applies on the way back:",
                "",
                "  +      pass gradient straight through to both inputs",
                "  *      scale gradient by the OTHER input's value",
                "  ^k     scale by k · x^(k−1)",
                "  tanh   scale by 1 − tanh(x)²",
                "  relu   pass it only where the input was positive, else block",
                "  exp    scale by the output itself (eˣ)"
        ));
        pages.add(String.join("\n",
                "Worked example:  (a*b) + c   with a=2, b=−3, c=10",
                "",
                "  forward:  a*b = −6,  (a*b)+c = 4",
                "  backward: dL/dc = 1            (+ passes through)",
                "            dL/d(a*b) = 1        (+ passes through)",
                "            dL/da = b = −3       (* scales by the other input)",
                "            dL/db = a = 2",
                "",
                "Try it: open the Autograd playground, assign a/b/c, type (a*b)+c, then :step."
        ));
        return pages;
    }
}
