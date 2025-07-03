import curses
import random
import time

WIDTH = 10
HEIGHT = 20

SHAPES = {
    'I': [[(0, 1), (1, 1), (2, 1), (3, 1)],
          [(1, 0), (1, 1), (1, 2), (1, 3)]],
    'J': [[(0, 0), (0, 1), (1, 1), (2, 1)],
          [(1, 0), (2, 0), (1, 1), (1, 2)],
          [(0, 1), (1, 1), (2, 1), (2, 2)],
          [(1, 0), (1, 1), (0, 2), (1, 2)]],
    'L': [[(2, 0), (0, 1), (1, 1), (2, 1)],
          [(1, 0), (1, 1), (1, 2), (2, 2)],
          [(0, 1), (1, 1), (2, 1), (0, 2)],
          [(0, 0), (1, 0), (1, 1), (1, 2)]],
    'O': [[(1, 0), (2, 0), (1, 1), (2, 1)]],
    'S': [[(1, 0), (2, 0), (0, 1), (1, 1)],
          [(1, 0), (1, 1), (2, 1), (2, 2)]],
    'T': [[(1, 0), (0, 1), (1, 1), (2, 1)],
          [(1, 0), (1, 1), (2, 1), (1, 2)],
          [(0, 1), (1, 1), (2, 1), (1, 2)],
          [(1, 0), (0, 1), (1, 1), (1, 2)]],
    'Z': [[(0, 0), (1, 0), (1, 1), (2, 1)],
          [(2, 0), (1, 1), (2, 1), (1, 2)]]
}


def create_board():
    return [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]


def rotate(shape, rotation):
    options = SHAPES[shape]
    return options[rotation % len(options)]


def valid(board, shape, offx, offy):
    for x, y in shape:
        nx, ny = x + offx, y + offy
        if nx < 0 or nx >= WIDTH or ny < 0 or ny >= HEIGHT:
            return False
        if board[ny][nx]:
            return False
    return True


def merge(board, shape, offx, offy, val):
    for x, y in shape:
        board[y + offy][x + offx] = val


def clear_lines(board):
    new_board = [row for row in board if any(v == 0 for v in row)]
    cleared = HEIGHT - len(new_board)
    for _ in range(cleared):
        new_board.insert(0, [0] * WIDTH)
    return new_board, cleared


def draw(stdscr, board, shape, offx, offy, score):
    stdscr.clear()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            ch = '#' if board[y][x] else '.'
            stdscr.addch(y, x, ch)
    for x, y in shape:
        stdscr.addch(offy + y, offx + x, '#')
    stdscr.addstr(0, WIDTH + 2, f"Score: {score}")
    stdscr.refresh()


def game(stdscr):
    curses.curs_set(0)
    board = create_board()
    score = 0
    key = None
    piece = random.choice(list(SHAPES.keys()))
    rotation = 0
    offx, offy = WIDTH // 2 - 2, 0
    fall_time = time.time()
    while True:
        shape = rotate(piece, rotation)
        if not valid(board, shape, offx, offy):
            break
        draw(stdscr, board, shape, offx, offy, score)
        timeout = 100
        stdscr.timeout(timeout)
        try:
            key = stdscr.getkey()
        except Exception:
            key = None
        if key == 'q':
            break
        elif key == 'KEY_LEFT' and valid(board, shape, offx - 1, offy):
            offx -= 1
        elif key == 'KEY_RIGHT' and valid(board, shape, offx + 1, offy):
            offx += 1
        elif key == 'KEY_UP':
            nr = rotation + 1
            if valid(board, rotate(piece, nr), offx, offy):
                rotation = nr
        elif key == 'KEY_DOWN' and valid(board, shape, offx, offy + 1):
            offy += 1
        if time.time() - fall_time > 0.5:
            if valid(board, shape, offx, offy + 1):
                offy += 1
            else:
                merge(board, shape, offx, offy, 1)
                board, cleared = clear_lines(board)
                score += cleared * 100
                piece = random.choice(list(SHAPES.keys()))
                rotation = 0
                offx, offy = WIDTH // 2 - 2, 0
                if not valid(board, rotate(piece, rotation), offx, offy):
                    break
            fall_time = time.time()
    stdscr.addstr(HEIGHT // 2, WIDTH // 2 - 4, "GAME OVER")
    stdscr.refresh()
    stdscr.getch()


if __name__ == "__main__":
    curses.wrapper(game)
