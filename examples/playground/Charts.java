import io.github.anandkrishanu.micrograd.MLP;

import java.util.List;

/** Plain-terminal visualizations: a loss line-chart, a sparkline, and a 2D decision boundary. */
final class Charts {
    private Charts() {}

    /** Line chart of {@code loss} over steps, auto-scaled, sized to {@code width} x {@code height}. */
    static String lossCurve(List<Double> loss, int width, int height) {
        if (loss.isEmpty()) return "  (no data yet)\n";
        int n = loss.size();
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : loss) {
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
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

    /** Compact one-line sparkline of {@code vals}, sampled to {@code width} cells. */
    static String sparkline(List<Double> vals, int width) {
        if (vals.isEmpty()) return "";
        char[] ticks = "▁▂▃▄▅▆▇█".toCharArray();
        double lo = Double.MAX_VALUE, hi = -Double.MAX_VALUE;
        for (double v : vals) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
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

    /**
     * 2D decision boundary: classify a grid by sign(model.forward), overlay the data points.
     * '#' = model says +,  '.' = model says -,  'O' = label +1,  'X' = label -1.
     */
    static String decisionBoundary(MLP model, double[][] xs, double[] ys, int cols, int rows) {
        double xmin = Double.MAX_VALUE, xmax = -Double.MAX_VALUE;
        double ymin = Double.MAX_VALUE, ymax = -Double.MAX_VALUE;
        for (double[] p : xs) {
            xmin = Math.min(xmin, p[0]);
            xmax = Math.max(xmax, p[0]);
            ymin = Math.min(ymin, p[1]);
            ymax = Math.max(ymax, p[1]);
        }
        double padX = (xmax - xmin) * 0.15 + 1e-6;
        double padY = (ymax - ymin) * 0.15 + 1e-6;
        xmin -= padX; xmax += padX; ymin -= padY; ymax += padY;

        String[][] cell = new String[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double x = xmin + (xmax - xmin) * c / (cols - 1);
                double y = ymax - (ymax - ymin) * r / (rows - 1);
                double pred = model.forward(new double[]{x, y}).data;
                cell[r][c] = pred > 0 ? Tui.color("#", Tui.GREEN) : Tui.color(".", Tui.RED);
            }
        }
        // overlay data points
        for (int i = 0; i < xs.length; i++) {
            int c = (int) Math.round((xs[i][0] - xmin) / (xmax - xmin) * (cols - 1));
            int r = (int) Math.round((ymax - xs[i][1]) / (ymax - ymin) * (rows - 1));
            if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
            cell[r][c] = ys[i] > 0 ? Tui.color("O", Tui.BOLD + Tui.GREEN)
                                   : Tui.color("X", Tui.BOLD + Tui.RED);
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
        sb.append("  ").append(Tui.color("# +region   . -region   O class+1   X class-1", Tui.DIM)).append('\n');
        return sb.toString();
    }
}
