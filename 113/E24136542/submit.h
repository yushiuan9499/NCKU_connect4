// File Name can be anything.
// DON'T change the function name and arguments.

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;

#pragma GCC optimize("O2")
// ===============  ToDo: 請將你的學號命名你的namespace
namespace E24136542
// ===================================================
{
const int tryMove[7] = {3, 2, 4, 1, 5, 0, 6};
struct Entry {
  uint64_t key : 56;
  int value;
};

class CacheTable {
public:
  CacheTable() { memset(table, 0, sizeof(table)); }
  void put(uint64_t key, int value) {
    Entry &e = table[key % sz];
    e.key = key; // this may clear the previous value
    e.value = value;
  }
  int get(uint64_t key) {
    Entry &e = table[key % sz];
    if (e.key == key) // check if the key is the same
      return e.value;
    else
      return 0;
  }

private:
  const static int sz = 52428767; // about 400MB
  struct Entry {
    uint64_t key : 56;
    int value;
  };
  Entry table[sz];
} cache;
constexpr static unsigned long long bottom(int width, int height) {
  return width == 0
             ? 0
             : bottom(width - 1, height) | 1LL << (width - 1) * (height + 1);
}
class Position {
public:
  static const int WIDTH = 7;  // width of the board
  static const int HEIGHT = 6; // height of the board
  static const int MIN_SCORE = -(WIDTH * HEIGHT) / 2 + 3;

  void play(uint64_t move) {
    currentPosition ^= mask;
    mask |= move;
    moves++;
  }
  void playCol(int col) {
    play((mask + BOTTOM_MASK_COL(col)) & COLUMN_MASK(col));
  }

  bool canWinNext() const { return winningPosition() & possible(); }

  int getMoves() const { return moves; }

  // this key is the unique
  uint64_t key() const { return currentPosition + mask; }

  uint64_t possibleNonLosingMoves() const {
    uint64_t possible_mask = possible();
    uint64_t opponent_win = opponentWinningPosition();
    uint64_t forced_moves = possible_mask & opponent_win;
    if (forced_moves) {
      if (forced_moves &
          (forced_moves - 1)) // check if there is more than one forced move
        return 0; // the opponnent has two winning moves and you cannot stop him
      else
        possible_mask = forced_moves; // enforce to play the single forced move
    }
    return possible_mask &
           ~(opponent_win >> 1); // avoid to play below an opponent winning spot
  }

  int moveScore(uint64_t move) const {
    return popcount(computeWinningPosition(currentPosition | move, mask));
  }
  int posScore(uint64_t next) const {
    int ret = 0;
    for (int i = Position::WIDTH; i--;)
      if (uint64_t move = next & Position::COLUMN_MASK(i))
        ret = max(ret, moveScore(move));
    return ret;
  }

  Position() : currentPosition{0}, mask{0}, moves{0} {}
  Position(const vector<vector<char>> &b, char me, char opponent) {
    currentPosition = 0;
    mask = 0;
    moves = 0;
    for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
        if (b[5 - i][j] == me) {
          currentPosition |= 1ll << (i + j * (HEIGHT + 1));
          mask |= currentPosition;
        } else if (b[5 - i][j] == opponent) {
          mask |= 1ll << (i + j * (HEIGHT + 1));
        }
      }
    }
    moves = popcount(mask);
  }

  uint64_t currentPosition; // bitmap of the current_player stones
  uint64_t mask;            // bitmap of all the already palyed spots
  unsigned int moves; // number of moves played since the beinning of the game.

  bool canPlay(int col) const { return (mask & TOP_MASK_COL(col)) == 0; }

  bool isWinningMove(int col) const {
    return winningPosition() & possible() & COLUMN_MASK(col);
  }

  // get all place that can win
  uint64_t winningPosition() const {
    return computeWinningPosition(currentPosition, mask);
  }

  uint64_t opponentWinningPosition() const {
    return computeWinningPosition(currentPosition ^ mask, mask);
  }

  // all possible moves
  uint64_t possible() const { return (mask + BOTTOM_MASK) & BOARD_MASK; }

  static unsigned int popcount(uint64_t m) {
    unsigned int c = 0;
    for (c = 0; m; c++)
      m &= m - 1;
    return c;
  }

  static uint64_t computeWinningPosition(uint64_t position, uint64_t mask) {
    // vertical;
    uint64_t r = (position << 1) & (position << 2) & (position << 3);

    // horizontal
    uint64_t p = (position << (HEIGHT + 1)) & (position << 2 * (HEIGHT + 1));
    r |= p & (position << 3 * (HEIGHT + 1)); //.***
    r |= p & (position >> (HEIGHT + 1));     //*.**
    p = (position >> (HEIGHT + 1)) & (position >> 2 * (HEIGHT + 1));
    r |= p & (position << (HEIGHT + 1));     //**.*
    r |= p & (position >> 3 * (HEIGHT + 1)); //***.

    // diagonal 1
    p = (position << HEIGHT) & (position << 2 * HEIGHT);
    r |= p & (position << 3 * HEIGHT);
    r |= p & (position >> HEIGHT);
    p = (position >> HEIGHT) & (position >> 2 * HEIGHT);
    r |= p & (position << HEIGHT);
    r |= p & (position >> 3 * HEIGHT);

    // diagonal 2
    p = (position << (HEIGHT + 2)) & (position << 2 * (HEIGHT + 2));
    r |= p & (position << 3 * (HEIGHT + 2));
    r |= p & (position >> (HEIGHT + 2));
    p = (position >> (HEIGHT + 2)) & (position >> 2 * (HEIGHT + 2));
    r |= p & (position << (HEIGHT + 2));
    r |= p & (position >> 3 * (HEIGHT + 2));

    return r & (BOARD_MASK ^ mask);
  }

  // Static bitmaps

  const static uint64_t BOTTOM_MASK = bottom(WIDTH, HEIGHT);
  const static uint64_t BOARD_MASK = BOTTOM_MASK * ((1LL << HEIGHT) - 1);

  // return a bitmask containg a single 1 corresponding to the top cel of a
  // given column
  static constexpr uint64_t TOP_MASK_COL(int col) {
    return UINT64_C(1) << ((HEIGHT - 1) + col * (HEIGHT + 1));
  }

  // return a bitmask containg a single 1 corresponding to the bottom cell of a
  // given column
  static constexpr uint64_t BOTTOM_MASK_COL(int col) {
    return UINT64_C(1) << col * (HEIGHT + 1);
  }

public:
  // return a bitmask 1 on all the cells of a given column
  static constexpr uint64_t COLUMN_MASK(int col) {
    return ((UINT64_C(1) << HEIGHT) - 1) << col * (HEIGHT + 1);
  }
};
class MoveSorter {
public:
  // add a move to the list
  // and keep the list sorted by score
  void add(uint64_t move, int score) {
    int pos = size++;
    for (; pos && entries[pos - 1].score > score; --pos)
      entries[pos] = entries[pos - 1];
    entries[pos].move = move;
    entries[pos].score = score;
  }

  uint64_t getNext() {
    if (size)
      return entries[--size].move;
    else
      return 0;
  }

  MoveSorter() : size{0} {}

private:
  // number of stored moves
  unsigned int size;

  // Contains size moves with their score ordered by score
  struct {
    uint64_t move;
    int score;
  } entries[Position::WIDTH];
};
int restrictNegamax(const Position &P, int alpha, int beta, int depth = 9) {
  uint64_t next = P.possibleNonLosingMoves();
  if (!next) // if no possible non losing move, opponent wins next move
    return -(Position::WIDTH * Position::HEIGHT - P.getMoves()) / 2;
  int mn = -(Position::WIDTH * Position::HEIGHT - 2 - P.getMoves()) /
           2; // lower bound of score as opponent cannot win next move
  if (alpha < mn) {
    alpha = mn; // there is no need to keep beta above our mx possible score.
    if (alpha >= beta)
      return alpha; // prune the exploration if the [alpha;beta] window is
                    // empty.
  }
  int mx = (Position::WIDTH * Position::HEIGHT - 1 - P.getMoves()) /
           2; // upper bound of our score as we cannot win immediately
  if (int ret = cache.get(P.key()))
    mx = ret;
  if (beta > mx) {
    beta = mx; // there is no need to keep beta above our max possible score.
    if (alpha >= beta)
      return beta; // prune the exploration if the [alpha;beta] window is empty.
  }

  if (!depth) {
    return min(mx, P.posScore(next));
  }
  MoveSorter moves;

  for (int i = Position::WIDTH; i--;)
    if (uint64_t move = next & Position::COLUMN_MASK(tryMove[i]))
      moves.add(move, P.moveScore(move));

  while (uint64_t next = moves.getNext()) {
    Position P2(P);
    P2.play(next); // It's opponent turn in P2 position after current player
                   // plays x column.
    int score = -restrictNegamax(P2, -beta, -alpha, depth - 1);

    if (score >= beta)
      return score; // prune the exploration if we find a possible move better
    // than what we were looking for.
    if (score > alpha)
      alpha = score;
  }

  if (alpha < 0 ||
      alpha >
          (Position::WIDTH * Position::HEIGHT - 1 - P.getMoves() - depth) / 2)
    cache.put(P.key(), min(mx, alpha));
  return alpha;
}

} // namespace E24136542
int getMove(const vector<vector<char>> &b, char mydisc, char yourdisc) {
  using namespace E24136542;
  Position pos(b, mydisc, yourdisc);
  for (int i = 0; i < Position::WIDTH; ++i) {
    if (pos.isWinningMove(i))
      return i;
  }
  uint64_t next = pos.possibleNonLosingMoves();
  if (!next) {
    for (int i = 0; i < Position::WIDTH; ++i) {
      if (pos.canPlay(tryMove[i])) {
        return tryMove[i];
      }
    }
  }
  int best = -20, bestMove = 0;
  MoveSorter moves;

  for (int i = Position::WIDTH; i--;)
    if (uint64_t move = next & Position::COLUMN_MASK(tryMove[i]))
      moves.add(tryMove[i], pos.moveScore(move));

  while (uint64_t next = moves.getNext()) {
    Position newPos(pos);
    newPos.playCol(next); // try to play the move
    int tmp = -restrictNegamax(newPos, -20, -best);
    if (tmp > best) {
      best = tmp;
      bestMove = next;
    }
  }
  return bestMove;
}
