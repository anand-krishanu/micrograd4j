import org.jline.reader.EndOfFileException;
import org.jline.reader.LineReader;
import org.jline.reader.LineReaderBuilder;
import org.jline.reader.UserInterruptException;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;

import java.io.PrintWriter;
import java.util.function.Supplier;

/**
 * Thin wrapper around JLine that the rest of the playground talks to.
 *
 * Everything is static so call sites stay short. When there is no real terminal
 * (e.g. input is piped, or {@code --demo} is run under CI) JLine falls back to a
 * "dumb" terminal: we then drop ANSI codes, screen clears, animations and the
 * arrow-key menus so the output stays readable in a plain log.
 */
final class Tui {
    // ----- ANSI styling (suppressed on dumb terminals) -----
    static final String ESC = "\033";
    static final String RESET = ESC + "[0m";
    static final String BOLD = ESC + "[1m";
    static final String DIM = ESC + "[2m";
    static final String ITALIC = ESC + "[3m";
    static final String REVERSE = ESC + "[7m";
    static final String RED = ESC + "[31m";
    static final String GREEN = ESC + "[32m";
    static final String YELLOW = ESC + "[33m";
    static final String BLUE = ESC + "[34m";
    static final String MAGENTA = ESC + "[35m";
    static final String CYAN = ESC + "[36m";
    static final String GRAY = ESC + "[90m";

    // ----- theme (256-colour accents; degrade to basic colours when unsupported) -----
    static final int ACCENT_256 = 39;   // deep sky blue
    static final int MUTED_256 = 244;   // soft grey
    static final String ACCENT = CYAN;  // basic-colour stand-in

    // ----- in-place redraw helpers -----
    static final String CLR_EOL = ESC + "[K";    // erase to end of line
    static final String CLR_DOWN = ESC + "[0J";   // erase from cursor to end of screen
    static final String HOME = ESC + "[H";

    // ----- spinner -----
    private static final String[] SPINNER = {
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
    };

    static Terminal terminal;
    static LineReader reader;
    static PrintWriter out;
    static boolean dumb;
    static boolean color256;

    /** Live completion sources the playground feeds the line reader (e.g. variable names). */
    static Supplier<java.util.Collection<String>> dynamicWords = java.util.Collections::emptyList;

    private Tui() {}

    static void init() {
        try {
            terminal = TerminalBuilder.builder().system(true).dumb(true).build();
        } catch (Exception e) {
            throw new RuntimeException("Could not open a terminal: " + e.getMessage(), e);
        }
        String type = terminal.getType();
        dumb = type == null || type.contains("dumb");
        color256 = !dumb;
        out = terminal.writer();
        // completer: static keywords plus whatever the playground exposes live (e.g. variables)
        reader = LineReaderBuilder.builder()
                .terminal(terminal)
                .completer((rdr, line, candidates) -> {
                    for (String w : BASE_WORDS) {
                        candidates.add(new org.jline.reader.Candidate(w));
                    }
                    for (String w : dynamicWords.get()) {
                        candidates.add(new org.jline.reader.Candidate(w));
                    }
                })
                .build();
    }

    private static final String[] BASE_WORDS = {
            "tanh(", "relu(", "exp(", ":graph", ":step", ":net", ":examples",
            ":explain", ":vars", ":help", ":back", "moons", "xor", "circles", "custom"
    };

    static void close() {
        try {
            if (terminal != null) terminal.close();
        } catch (Exception ignored) {
        }
    }

    static int width() {
        int w = terminal != null ? terminal.getWidth() : 0;
        return w > 0 ? w : 80;
    }

    static int height() {
        int h = terminal != null ? terminal.getHeight() : 0;
        return h > 0 ? h : 24;
    }

    // ----- output -----
    static String color(String s, String code) {
        return dumb ? s : code + s + RESET;
    }

    /** 256-colour foreground; degrades to the supplied basic-colour fallback, then to plain. */
    static String color256(String s, int code, String basicFallback) {
        if (dumb) return s;
        if (!color256) return basicFallback == null ? s : basicFallback + s + RESET;
        return ESC + "[38;5;" + code + "m" + s + RESET;
    }

    /** 256-colour background; degrades to plain on dumb terminals. */
    static String bg256(String s, int code) {
        if (dumb || !color256) return s;
        return ESC + "[48;5;" + code + "m" + s + RESET;
    }

    static void print(String s) {
        out.print(s);
    }

    static void println(String s) {
        out.println(s);
    }

    static void println() {
        out.println();
    }

    static void flush() {
        out.flush();
    }

    /** Clear the screen (or just separate sections on a dumb terminal). */
    static void clear() {
        if (dumb) {
            out.println("\n");
        } else {
            out.print(ESC + "[2J" + ESC + "[3J" + HOME);
        }
        out.flush();
    }

    /** Move the cursor home without wiping — used to redraw animations with less flicker. */
    static void home() {
        if (!dumb) {
            out.print(HOME);
            out.flush();
        }
    }

    /** Erase from the cursor to the end of the screen (clears stale animation lines). */
    static void eraseDown() {
        if (!dumb) {
            out.print(CLR_DOWN);
            out.flush();
        }
    }

    // ----- visible width (ignores ANSI escapes) -----
    static int visibleLen(String s) {
        int n = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\033') {                       // skip CSI ... letter
                int j = i + 1;
                if (j < s.length() && s.charAt(j) == '[') {
                    j++;
                    while (j < s.length() && !Character.isLetter(s.charAt(j))) j++;
                }
                i = j;
            } else {
                n++;
            }
        }
        return n;
    }

    // ----- framing -----
    private static int boxWidth() {
        return Math.min(width(), 72);
    }

    /** Single-line titled box (kept for back-compat; now with rounded corners). */
    static void header(String title) {
        int w = boxWidth();
        String bar = "─".repeat(Math.max(0, w - 2));
        println(color("╭" + bar + "╮", ACCENT));
        String padded = " " + title;
        padded = padded + " ".repeat(Math.max(0, w - 1 - title.length() - 1));
        println(color("│", ACCENT) + color(padded, BOLD) + color("│", ACCENT));
        println(color("╰" + bar + "╯", ACCENT));
    }

    /** Open a panel: a title rule with the title inlined. Pair with {@link #panelBottom()}. */
    static void panelTop(String title) {
        int w = boxWidth();
        String label = " " + title + " ";
        int rest = Math.max(0, w - 2 - label.length() - 2);
        println(color("╭─", ACCENT) + color(label, BOLD) + color("─".repeat(rest) + "─╮", ACCENT));
    }

    static void panelBottom() {
        int w = boxWidth();
        println(color("╰" + "─".repeat(Math.max(0, w - 2)) + "╯", ACCENT));
    }

    /** A bordered panel containing pre-formatted (possibly coloured) body lines. */
    static void panel(String title, String body) {
        panelTop(title);
        for (String line : body.split("\n", -1)) {
            if (line.isEmpty()) { println(); continue; }
            println("  " + line);
        }
        panelBottom();
    }

    /** Dim status line, e.g. key hints at the bottom of a screen. */
    static void footer(String hint) {
        println(color("  " + hint, DIM));
    }

    // ----- widgets -----
    /** Horizontal progress bar built from block glyphs; coloured unless dumb. */
    static String progressBar(double frac, int barWidth) {
        frac = Math.max(0.0, Math.min(1.0, frac));
        int full = (int) Math.round(frac * barWidth);
        String filled = "█".repeat(full);
        String empty = "░".repeat(Math.max(0, barWidth - full));
        if (dumb) return filled + empty;
        return color256(filled, ACCENT_256, ACCENT) + color(empty, GRAY);
    }

    /** One spinner frame for the given tick. */
    static String spinner(int tick) {
        return SPINNER[Math.floorMod(tick, SPINNER.length)];
    }

    // ----- input -----
    /** Read one line; returns null on EOF / Ctrl-C / Ctrl-D (treated as "go back"). */
    static String readLine(String prompt) {
        try {
            return reader.readLine(prompt);
        } catch (EndOfFileException | UserInterruptException e) {
            return null;
        }
    }

    static double readDouble(String prompt, double def) {
        while (true) {
            String s = readLine(prompt + color(" [" + trim(def) + "]: ", DIM));
            if (s == null) return def;
            s = s.trim();
            if (s.isEmpty()) return def;
            try {
                return Double.parseDouble(s);
            } catch (NumberFormatException e) {
                println(color("  not a number, try again", RED));
            }
        }
    }

    static int readInt(String prompt, int def) {
        while (true) {
            String s = readLine(prompt + color(" [" + def + "]: ", DIM));
            if (s == null) return def;
            s = s.trim();
            if (s.isEmpty()) return def;
            try {
                return Integer.parseInt(s);
            } catch (NumberFormatException e) {
                println(color("  not a whole number, try again", RED));
            }
        }
    }

    private static String trim(double d) {
        if (d == Math.rint(d) && !Double.isInfinite(d)) return String.valueOf((long) d);
        return String.valueOf(d);
    }

    static void pressEnter() {
        readLine(color("  (press Enter to continue) ", DIM));
    }
}
