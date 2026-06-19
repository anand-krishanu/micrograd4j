/**
 * A monochrome plotting canvas backed by Unicode braille (U+2800..U+28FF). Each character cell
 * packs a 2×4 grid of dots, so the effective resolution is {@code 2·cols × 4·rows} — enough to
 * draw a smooth, continuous line in a terminal.
 *
 * Dot numbering (standard braille):
 * <pre>
 *   (0,0)=1  (1,0)=4      bits: 1=0x01 2=0x02 3=0x04 4=0x08
 *   (0,1)=2  (1,1)=5            5=0x10 6=0x20 7=0x40 8=0x80
 *   (0,2)=3  (1,2)=6
 *   (0,3)=7  (1,3)=8
 * </pre>
 */
final class Braille {
    private final int cols, rows;
    private final int dotW, dotH;
    private final int[][] cell;   // [rows][cols] bitmask of lit dots

    Braille(int cols, int rows) {
        this.cols = Math.max(1, cols);
        this.rows = Math.max(1, rows);
        this.dotW = this.cols * 2;
        this.dotH = this.rows * 4;
        this.cell = new int[this.rows][this.cols];
    }

    int dotWidth() { return dotW; }
    int dotHeight() { return dotH; }
    int rows() { return rows; }

    void set(int x, int y) {
        if (x < 0 || y < 0 || x >= dotW || y >= dotH) return;
        cell[y / 4][x / 2] |= bit(x % 2, y % 4);
    }

    private static int bit(int dx, int dy) {
        if (dy < 3) return dx == 0 ? (1 << dy) : (8 << dy);   // 0x01/02/04  |  0x08/10/20
        return dx == 0 ? 0x40 : 0x80;
    }

    /** Bresenham line between two dot coordinates. */
    void line(int x0, int y0, int x1, int y1) {
        int dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;
        while (true) {
            set(x0, y0);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }

    /** One string per character row (top to bottom). */
    String[] render() {
        String[] out = new String[rows];
        for (int r = 0; r < rows; r++) {
            StringBuilder sb = new StringBuilder(cols);
            for (int c = 0; c < cols; c++) sb.append((char) (0x2800 + cell[r][c]));
            out[r] = sb.toString();
        }
        return out;
    }
}
