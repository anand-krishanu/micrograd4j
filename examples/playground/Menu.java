import org.jline.keymap.BindingReader;
import org.jline.keymap.KeyMap;
import org.jline.terminal.Attributes;
import org.jline.utils.InfoCmp.Capability;

import java.util.ArrayList;
import java.util.List;

import static org.jline.keymap.KeyMap.esc;
import static org.jline.keymap.KeyMap.key;

/**
 * An arrow-key selection menu. On a real terminal it draws a highlighted list you move through
 * with ↑/↓ (or j/k), pick with Enter, and leave with q/Esc; digits 1-9 jump straight to an item.
 *
 * On a dumb terminal (piped input, {@code --demo}) it degrades to a plain numbered prompt so the
 * playground keeps working in logs and CI.
 */
final class Menu {
    /** One selectable row: a {@code label} and a short dim {@code hint}. */
    static final class Item {
        final String label;
        final String hint;
        Item(String label, String hint) { this.label = label; this.hint = hint; }
        static Item of(String label, String hint) { return new Item(label, hint); }
    }

    private enum Op { UP, DOWN, SELECT, QUIT, DIGIT, NONE }

    private Menu() {}

    /**
     * Show {@code items} under {@code title}; return the chosen index, or -1 for back/quit.
     * {@code subtitle} (nullable) is a dim line under the title; {@code footer} (nullable)
     * overrides the default key-hint line.
     */
    static int select(String title, String subtitle, List<Item> items, String footer) {
        return Tui.dumb ? selectDumb(title, items) : selectRich(title, subtitle, items, footer);
    }

    static int select(String title, List<Item> items) {
        return select(title, null, items, null);
    }

    // ----- rich (arrow-key) path -----
    private static int selectRich(String title, String subtitle, List<Item> items, String footer) {
        int n = items.size();
        String hint = footer != null ? footer
                : "↑/↓ move · Enter select · 1-" + Math.min(9, n) + " jump · q back";

        KeyMap<Op> km = new KeyMap<>();
        bind(km, Op.UP, key(Tui.terminal, Capability.key_up), "\033[A", "\033OA");
        bind(km, Op.DOWN, key(Tui.terminal, Capability.key_down), "\033[B", "\033OB");
        bind(km, Op.UP, "k");
        bind(km, Op.DOWN, "j");
        bind(km, Op.SELECT, "\r", "\n");
        bind(km, Op.QUIT, "q", "Q", esc());
        for (int i = 1; i <= Math.min(9, n); i++) bind(km, Op.DIGIT, String.valueOf(i));
        km.setNomatch(Op.NONE);
        km.setAmbiguousTimeout(50L);   // lone Esc resolves fast; arrow sequences still arrive in time

        BindingReader br = new BindingReader(Tui.terminal.reader());
        Attributes saved = Tui.terminal.enterRawMode();
        int sel = 0;
        try {
            Tui.clear();
            render(title, subtitle, items, hint, sel);
            while (true) {
                Op op = br.readBinding(km);
                if (op == null) return -1;            // EOF
                switch (op) {
                    case UP:     sel = (sel - 1 + n) % n; break;
                    case DOWN:   sel = (sel + 1) % n; break;
                    case SELECT: return sel;
                    case QUIT:   return -1;
                    case DIGIT:
                        int d = br.getLastBinding().charAt(0) - '1';
                        if (d >= 0 && d < n) return d;
                        continue;
                    default:
                        continue;                      // unknown key: no redraw needed
                }
                render(title, subtitle, items, hint, sel);
            }
        } finally {
            Tui.terminal.setAttributes(saved);
            Tui.print("\r\n");
            Tui.flush();
        }
    }

    /** Redraw in place. In raw mode the terminal does no \n→\r\n translation, so emit \r\n. */
    private static void render(String title, String subtitle, List<Item> items, String hint, int sel) {
        StringBuilder sb = new StringBuilder();
        sb.append(Tui.HOME);
        for (String t : title.split("\n", -1)) sb.append(line(Tui.color("  " + t, Tui.BOLD + Tui.CYAN)));
        if (subtitle != null) sb.append(line(Tui.color("  " + subtitle, Tui.DIM)));
        sb.append(line(""));
        for (int i = 0; i < items.size(); i++) {
            Item it = items.get(i);
            boolean on = i == sel;
            String marker = on ? Tui.color("❯ ", Tui.BOLD + Tui.CYAN) : "  ";
            String num = Tui.color((i + 1) + ".", Tui.DIM);
            String label = on ? Tui.color(it.label, Tui.BOLD) : it.label;
            String tail = it.hint == null || it.hint.isEmpty() ? "" : Tui.color("  " + it.hint, Tui.DIM);
            sb.append(line("  " + marker + num + " " + label + tail));
        }
        sb.append(line(""));
        sb.append(line(Tui.color("  " + hint, Tui.DIM)));
        sb.append(Tui.CLR_DOWN);
        Tui.print(sb.toString());
        Tui.flush();
    }

    private static String line(String s) {
        return s + Tui.CLR_EOL + "\r\n";
    }

    // ----- dumb (typed) path -----
    private static int selectDumb(String title, List<Item> items) {
        Tui.println();
        String flatTitle = title.replace("\n", " — ");
        Tui.println(Tui.color("== " + flatTitle + " ==", Tui.BOLD));
        for (int i = 0; i < items.size(); i++) {
            Item it = items.get(i);
            String tail = it.hint == null || it.hint.isEmpty() ? "" : "  — " + it.hint;
            Tui.println("  " + (i + 1) + ") " + it.label + tail);
        }
        String c = Tui.readLine("\nchoose [1-" + items.size() + ", q]: ");
        if (c == null) return -1;
        c = c.trim().toLowerCase();
        if (c.isEmpty() || c.equals("q") || c.equals("quit") || c.equals("exit") || c.equals("b") || c.equals("back")) {
            return -1;
        }
        try {
            int idx = Integer.parseInt(c) - 1;
            if (idx >= 0 && idx < items.size()) return idx;
        } catch (NumberFormatException ignored) {
        }
        return -2;   // unrecognised: caller loops and re-renders
    }

    // ----- helpers -----
    private static void bind(KeyMap<Op> km, Op op, CharSequence... seqs) {
        List<CharSequence> ok = new ArrayList<>();
        for (CharSequence s : seqs) if (s != null && s.length() > 0) ok.add(s);
        if (!ok.isEmpty()) km.bind(op, ok.toArray(new CharSequence[0]));
    }
}
