import org.jline.reader.EndOfFileException;
import org.jline.reader.LineReader;
import org.jline.reader.LineReaderBuilder;
import org.jline.reader.UserInterruptException;
import org.jline.reader.impl.completer.StringsCompleter;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;

import java.io.PrintWriter;

/**
 * Thin wrapper around JLine that the rest of the playground talks to.
 *
 * Everything is static so call sites stay short. When there is no real terminal
 * (e.g. input is piped, or {@code --demo} is run under CI) JLine falls back to a
 * "dumb" terminal: we then drop ANSI codes and screen clears so the output stays
 * readable in a plain log.
 */
final class Tui {
    // ----- ANSI styling (suppressed on dumb terminals) -----
    static final String RESET = "[0m";
    static final String BOLD = "[1m";
    static final String DIM = "[2m";
    static final String RED = "[31m";
    static final String GREEN = "[32m";
    static final String YELLOW = "[33m";
    static final String BLUE = "[34m";
    static final String MAGENTA = "[35m";
    static final String CYAN = "[36m";
    static final String GRAY = "[90m";

    static Terminal terminal;
    static LineReader reader;
    static PrintWriter out;
    static boolean dumb;

    private Tui() {}

    static void init() {
        try {
            terminal = TerminalBuilder.builder().system(true).dumb(true).build();
        } catch (Exception e) {
            throw new RuntimeException("Could not open a terminal: " + e.getMessage(), e);
        }
        String type = terminal.getType();
        dumb = type == null || type.contains("dumb");
        out = terminal.writer();
        reader = LineReaderBuilder.builder()
                .terminal(terminal)
                .completer(new StringsCompleter(
                        "tanh(", "relu(", "exp(", ":graph", ":step", ":net",
                        ":help", ":back", "moons", "xor", "circles", "custom"))
                .build();
    }

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
            out.print("[2J[3J[H");
        }
        out.flush();
    }

    /** Move the cursor home without wiping — used to redraw animations with less flicker. */
    static void home() {
        if (!dumb) {
            out.print("[H");
            out.flush();
        }
    }

    static void header(String title) {
        int w = Math.min(width(), 72);
        String bar = "─".repeat(Math.max(0, w - 2));
        println(color("┌" + bar + "┐", CYAN));
        String padded = " " + title;
        padded = padded + " ".repeat(Math.max(0, w - 1 - title.length() - 1));
        println(color("│", CYAN) + color(padded, BOLD) + color("│", CYAN));
        println(color("└" + bar + "┘", CYAN));
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
            String s = readLine(prompt + " [" + def + "]: ");
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
            String s = readLine(prompt + " [" + def + "]: ");
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

    static void pressEnter() {
        readLine(color("  (press Enter to continue) ", DIM));
    }
}
