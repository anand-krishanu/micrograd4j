import io.github.anandkrishanu.micrograd.MLP;

import java.util.List;

/**
 * Terminal visualizations: a braille loss curve, a sparkline, and a heatmap decision boundary.
 * Each "rich" rendering has an ASCII fallback used on dumb terminals (piped input / {@code --demo})
 * so the playground stays readable in plain logs.
 */
final class Charts {
    private Charts() {}

    // hot→cool vertical ramp for the loss line (top = high loss = warm)
    private static final int[] LOSS_RAMP = {196, 202, 208, 214, 220, 190, 154, 118, 82, 46};
    // confidence ramps for the decision-boundary heatmap (dim → vivid)
    private static final int[] GREENS = {22, 28, 34, 40, 46};
    private static final int[] REDS = {52, 88, 124, 160, 196};

    // ---------------------------------------------------------------- loss curve
    /** Line chart of {@code loss} over steps, auto-scaled, sized to {@code width} x {@code height}. */
    static String lossCurve(List<Double> loss, int width, int height) {
        if (loss.isEmpty()) return "  (no data yet)\n";
        return Tui.dumb ? lossCurveAscii(loss, width, height)
                        : lossCurveBraille(loss, width, height);
    }

    private static String lossCurveBraille(List<Double> loss, int width, int height) {
        int n = loss.size();
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : loss) { min = Math.min(min, v); max = Math.max(max, v); }
        double range = max - min;
        if (range < 1e-12) range = 1.0;

        Braille canvas = new Braille(width, height);
        int dw = canvas.dotWidth(), dh = canvas.dotHeight();
        int prevX = -1, prevY = -1;
        for (int i = 0; i < n; i++) {
            int x = n == 1 ? 0 : (int) Math.round((double) i / (n - 1) * (dw - 1));
            int y = (int) Math.round((1 - (loss.get(i) - min) / range) * (dh - 1));
            y = Math.max(0, Math.min(dh - 1, y));
            if (prevX >= 0) canvas.line(prevX, prevY, x, y); else canvas.set(x, y);
            prevX = x; prevY = y;
        }

        String[] grid = canvas.render();
        StringBuilder sb = new StringBuilder();
        int rows = grid.length;
        for (int r = 0; r < rows; r++) {
            String yLabel = r == 0 ? String.format("%8.4f", max)
                    : r == rows - 1 ? String.format("%8.4f", min)
                    : r == rows / 2 ? String.format("%8.4f", min + range / 2)
                    : "        ";
            int rampIdx = rows == 1 ? 0 : (int) Math.round((double) r / (rows - 1) * (LOSS_RAMP.length - 1));
            sb.append(yLabel).append(' ').append(Tui.color("┤", Tui.GRAY))
              .append(Tui.color256(grid[r], LOSS_RAMP[rampIdx], Tui.CYAN))
              .append('\n');
        }
        sb.append("         ").append(Tui.color("└" + "─".repeat(width), Tui.GRAY)).append('\n');
        sb.append("         ").append(Tui.color("step 0", Tui.DIM))
          .append(" ".repeat(Math.max(1, width - 12)))
          .append(Tui.color("step " + (n - 1), Tui.DIM)).append('\n');
        return sb.toString();
    }

    /** Original dot-grid chart; used as the dumb-terminal fallback. */
    private static String lossCurveAscii(List<Double> loss, int width, int height) {
        int n = loss.size();
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : loss) { min = Math.min(min, v); max = Math.max(max, v); }
        double range = max - min;
        if (range < 1e-12) range = 1.0;

        char[][] grid = new char[height][width];
        for (char[] row : grid) java.util.Arrays.fill(row, ' ');
        for (int c = 0; c < width; c++) {
            int idx = width == 1 ? n - 1 : (int) Math.round((double) c / (width - 1) * (n - 1));
            double v = loss.get(idx);
            int r = (int) Math.round((1 - (v - min) / range) * (height - 1));
            r = Math.max(0, Math.min(height - 1, r));
            grid[r][c] = '*';
        }

        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < height; r++) {
            String yLabel = r == 0 ? String.format("%7.4f", max)
                    : r == height - 1 ? String.format("%7.4f", min)
                    : "       ";
            sb.append(yLabel).append(" ").append(Tui.color("┤", Tui.GRAY));
            StringBuilder line = new StringBuilder();
            for (int c = 0; c < width; c++) line.append(grid[r][c] == '*' ? '•' : ' ');
            sb.append(Tui.color(line.toString(), Tui.CYAN)).append('\n');
        }
        sb.append("        ").append(Tui.color("└" + "─".repeat(width), Tui.GRAY)).append('\n');
        sb.append("        ").append(Tui.color("step 0", Tui.DIM))
          .append(" ".repeat(Math.max(1, width - 11)))
          .append(Tui.color("step " + (n - 1), Tui.DIM)).append('\n');
        return sb.toString();
    }

    // ---------------------------------------------------------------- sparkline
    /** Compact one-line sparkline of {@code vals}, sampled to {@code width} cells. */
    static String sparkline(List<Double> vals, int width) {
        if (vals.isEmpty()) return "";
        char[] ticks = "▁▂▃▄▅▆▇█".toCharArray();
        double lo = Double.MAX_VALUE, hi = -Double.MAX_VALUE;
        for (double v : vals) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        if (hi - lo < 1e-12) hi = lo + 1;
        int n = vals.size();
        int w = Math.min(width, n);
        StringBuilder sb = new StringBuilder();
        for (int c = 0; c < w; c++) {
            int idx = w == 1 ? n - 1 : (int) Math.round((double) c / (w - 1) * (n - 1));
            int level = (int) Math.round((vals.get(idx) - lo) / (hi - lo) * (ticks.length - 1));
            sb.append(ticks[Math.max(0, Math.min(ticks.length - 1, level))]);
        }
        return sb.toString();
    }

    // ---------------------------------------------------------------- decision boundary
    /**
     * 2D decision boundary. Rich terminals get a confidence heatmap (green = model says +,
     * red = model says −, brighter = more confident) with data points overlaid; dumb terminals
     * get the original {@code #/./O/X} grid.
     */
    static String decisionBoundary(MLP model, double[][] xs, double[] ys, int cols, int rows) {
        return Tui.dumb ? decisionBoundaryAscii(model, xs, ys, cols, rows)
                        : decisionBoundaryHeatmap(model, xs, ys, cols, rows);
    }

    private static double[] bounds(double[][] xs) {
        double xmin = Double.MAX_VALUE, xmax = -Double.MAX_VALUE;
        double ymin = Double.MAX_VALUE, ymax = -Double.MAX_VALUE;
        for (double[] p : xs) {
            xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]);
            ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]);
        }
        double padX = (xmax - xmin) * 0.15 + 1e-6;
        double padY = (ymax - ymin) * 0.15 + 1e-6;
        return new double[]{xmin - padX, xmax + padX, ymin - padY, ymax + padY};
    }

    private static String decisionBoundaryHeatmap(MLP model, double[][] xs, double[] ys, int cols, int rows) {
        double[] b = bounds(xs);
        double xmin = b[0], xmax = b[1], ymin = b[2], ymax = b[3];

        String[][] cell = new String[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double x = xmin + (xmax - xmin) * c / (cols - 1);
                double y = ymax - (ymax - ymin) * r / (rows - 1);
                double pred = model.forward(new double[]{x, y}).data;
                int level = (int) Math.round(Math.tanh(Math.abs(pred)) * 4);
                level = Math.max(0, Math.min(4, level));
                int code = pred > 0 ? GREENS[level] : REDS[level];
                cell[r][c] = Tui.color256("█", code, pred > 0 ? Tui.GREEN : Tui.RED);
            }
        }
        // overlay data points: bright ● for class +1, bright ✕ for class −1
        for (int i = 0; i < xs.length; i++) {
            int c = (int) Math.round((xs[i][0] - xmin) / (xmax - xmin) * (cols - 1));
            int r = (int) Math.round((ymax - xs[i][1]) / (ymax - ymin) * (rows - 1));
            if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
            cell[r][c] = ys[i] > 0 ? Tui.color256("●", 231, Tui.BOLD)
                                   : Tui.color256("✕", 231, Tui.BOLD);
        }

        StringBuilder sb = new StringBuilder();
        String border = "─".repeat(cols);
        sb.append("  ").append(Tui.color("┌" + border + "┐", Tui.GRAY)).append('\n');
        for (int r = 0; r < rows; r++) {
            sb.append("  ").append(Tui.color("│", Tui.GRAY));
            for (int c = 0; c < cols; c++) sb.append(cell[r][c]);
            sb.append(Tui.color("│", Tui.GRAY)).append('\n');
        }
        sb.append("  ").append(Tui.color("└" + border + "┘", Tui.GRAY)).append('\n');
        sb.append("  ").append(Tui.color("● class +1   ✕ class −1   ", Tui.DIM))
          .append(Tui.color256("green", 46, Tui.GREEN)).append(Tui.color(" +region  ", Tui.DIM))
          .append(Tui.color256("red", 196, Tui.RED)).append(Tui.color(" −region   brighter = more confident", Tui.DIM))
          .append('\n');
        return sb.toString();
    }

    /** Original ASCII boundary; used as the dumb-terminal fallback. */
    private static String decisionBoundaryAscii(MLP model, double[][] xs, double[] ys, int cols, int rows) {
        double[] b = bounds(xs);
        double xmin = b[0], xmax = b[1], ymin = b[2], ymax = b[3];

        String[][] cell = new String[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double x = xmin + (xmax - xmin) * c / (cols - 1);
                double y = ymax - (ymax - ymin) * r / (rows - 1);
                double pred = model.forward(new double[]{x, y}).data;
                cell[r][c] = pred > 0 ? "#" : ".";
            }
        }
        for (int i = 0; i < xs.length; i++) {
            int c = (int) Math.round((xs[i][0] - xmin) / (xmax - xmin) * (cols - 1));
            int r = (int) Math.round((ymax - xs[i][1]) / (ymax - ymin) * (rows - 1));
            if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
            cell[r][c] = ys[i] > 0 ? "O" : "X";
        }

        StringBuilder sb = new StringBuilder();
        String border = "─".repeat(cols);
        sb.append("  ┌").append(border).append("┐\n");
        for (int r = 0; r < rows; r++) {
            sb.append("  │");
            for (int c = 0; c < cols; c++) sb.append(cell[r][c]);
            sb.append("│\n");
        }
        sb.append("  └").append(border).append("┘\n");
        sb.append("  # +region   . -region   O class+1   X class-1\n");
        return sb.toString();
    }
}
