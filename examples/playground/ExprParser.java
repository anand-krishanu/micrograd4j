import io.github.anandkrishanu.micrograd.Value;

import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A tiny recursive-descent parser that turns a typed expression into a micrograd
 * {@link Value} graph, using only the public ops on {@code Value}.
 *
 * Grammar (lowest to highest precedence):
 *   expr    := term (('+'|'-') term)*
 *   term    := unary (('*'|'/') unary)*
 *   unary   := '-' unary | power
 *   power   := postfix ('^' unary)?          (right-assoc; exponent taken as a constant)
 *   postfix := primary ('.' func '(' ')')*    e.g.  a.tanh()
 *   primary := number | ident | func '(' expr ')' | '(' expr ')'
 *   func    := tanh | relu | exp
 *
 * Named variables (assigned with {@code a = 2.0}) are reused as the SAME Value object
 * everywhere they appear, so the graph is a real DAG and gradients accumulate correctly.
 */
final class ExprParser {
    /** Variable name -> leaf Value, in insertion order (for stable display). */
    final Map<String, Value> vars = new LinkedHashMap<>();
    /** Value -> display label, by identity (constants and variables). */
    final Map<Value, String> names = new IdentityHashMap<>();

    private String src;
    private int pos;

    /** Create or overwrite a named leaf variable. */
    void assign(String name, double value) {
        Value v = new Value(value);
        vars.put(name, v);
        names.put(v, name);
    }

    /** Parse and build the graph for {@code expr}. Throws RuntimeException on a syntax error. */
    Value eval(String expr) {
        this.src = expr;
        this.pos = 0;
        Value v = parseExpr();
        skipWs();
        if (pos < src.length()) {
            throw new RuntimeException("unexpected '" + src.charAt(pos) + "' at column " + (pos + 1));
        }
        return v;
    }

    // ----- grammar -----
    private Value parseExpr() {
        Value left = parseTerm();
        while (true) {
            skipWs();
            char c = peek();
            if (c == '+') {
                pos++;
                left = left.add(parseTerm());
            } else if (c == '-') {
                pos++;
                left = left.subtract(parseTerm());
            } else {
                return left;
            }
        }
    }

    private Value parseTerm() {
        Value left = parseUnary();
        while (true) {
            skipWs();
            char c = peek();
            if (c == '*') {
                pos++;
                left = left.multiply(parseUnary());
            } else if (c == '/') {
                pos++;
                left = left.divide(parseUnary());
            } else {
                return left;
            }
        }
    }

    private Value parseUnary() {
        skipWs();
        if (peek() == '-') {
            pos++;
            return parseUnary().negate();
        }
        return parsePower();
    }

    private Value parsePower() {
        Value base = parsePostfix();
        skipWs();
        if (peek() == '^') {
            pos++;
            // exponent is a constant (not part of the graph), matching Value.power(double)
            Value exp = parseUnary();
            return base.power(exp.data);
        }
        return base;
    }

    private Value parsePostfix() {
        Value v = parsePrimary();
        while (true) {
            skipWs();
            if (peek() == '.') {
                int save = pos;
                pos++;
                String fn = readIdent();
                skipWs();
                if (fn.isEmpty() || peek() != '(') {
                    pos = save;
                    return v;
                }
                pos++; // '('
                skipWs();
                expect(')');
                v = applyFunc(fn, v);
            } else {
                return v;
            }
        }
    }

    private Value parsePrimary() {
        skipWs();
        char c = peek();
        if (c == '(') {
            pos++;
            Value v = parseExpr();
            skipWs();
            expect(')');
            return v;
        }
        if (Character.isDigit(c) || c == '.') {
            return number();
        }
        if (Character.isLetter(c)) {
            String id = readIdent();
            skipWs();
            if (peek() == '(' && isFunc(id)) {
                pos++;
                Value arg = parseExpr();
                skipWs();
                expect(')');
                return applyFunc(id, arg);
            }
            Value v = vars.get(id);
            if (v == null) {
                throw new RuntimeException("unknown variable '" + id + "' (assign it first, e.g. " + id + " = 1.0)");
            }
            return v;
        }
        throw new RuntimeException("unexpected '" + c + "' at column " + (pos + 1));
    }

    // ----- helpers -----
    private Value number() {
        int start = pos;
        while (pos < src.length() && (Character.isDigit(src.charAt(pos)) || src.charAt(pos) == '.')) pos++;
        // optional exponent
        if (pos < src.length() && (src.charAt(pos) == 'e' || src.charAt(pos) == 'E')) {
            pos++;
            if (pos < src.length() && (src.charAt(pos) == '+' || src.charAt(pos) == '-')) pos++;
            while (pos < src.length() && Character.isDigit(src.charAt(pos))) pos++;
        }
        double d = Double.parseDouble(src.substring(start, pos));
        Value v = new Value(d);
        names.put(v, trimNum(d));
        return v;
    }

    private String readIdent() {
        int start = pos;
        while (pos < src.length() && (Character.isLetterOrDigit(src.charAt(pos)) || src.charAt(pos) == '_')) pos++;
        return src.substring(start, pos);
    }

    private static boolean isFunc(String id) {
        return id.equals("tanh") || id.equals("relu") || id.equals("exp");
    }

    private static Value applyFunc(String fn, Value v) {
        switch (fn) {
            case "tanh": return v.tanh();
            case "relu": return v.relu();
            case "exp":  return v.exp();
            default: throw new RuntimeException("unknown function '" + fn + "' (use tanh, relu, exp)");
        }
    }

    private char peek() {
        return pos < src.length() ? src.charAt(pos) : '\0';
    }

    private void expect(char c) {
        if (peek() != c) {
            throw new RuntimeException("expected '" + c + "' at column " + (pos + 1));
        }
        pos++;
    }

    private void skipWs() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) pos++;
    }

    static String trimNum(double d) {
        if (d == Math.rint(d) && !Double.isInfinite(d)) {
            return String.valueOf((long) d);
        }
        return String.valueOf(d);
    }
}
