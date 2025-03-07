#include <cassert>
#include <iostream>
#include <cstdint>
#include <vector>
#include <array>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <ext/pb_ds/assoc_container.hpp>
using namespace std;

/*
 * main method
 * MINIMAX: minimax
 * PVS: pvs
 * MTDF_MINIMAX: mtdf + minimax
 * MTDF_PVS: mtdf + pvs
 * MTDF_IDDFS: iddfs + mtdf (need MTDF_MINIMAX or MTDF_PVS)
 * */

#define MINIMAX

/*
 * optimization
 * HISTORY: history heuristic
 * KILLER: killer heuristics
 * NMP: null move pruning
 * WTP: weird transposition table
 * CLEARTP: clear transposition table each match
 * LMR: late move reduction
 * DSE: dynamic search extension
 * STATICMOVEORDER: static column order strategy
 * */

#define CLEARTP

#ifdef TOBIICHI3227
#include <algo/debug.h>
#define debug(...) \
    cerr << "[" << __FUNCTION__ << "]L: " << __LINE__,\
    cerr << " (" << #__VA_ARGS__ << ") =", debug_out(__VA_ARGS__)
#else
#define debug(...) 3227
#endif

#define UNUSED(x) (void)(x)

namespace tobiichi2 {
static bool gIsZobristSeted = false;
static char player1Disc = '\0';
static char player2Disc = '\0';
static const int MAX_DEPTH = 9;
static const int NUM_COL = 7;
static const int NUM_ROW = 6;
using board_t = vector<vector<char>>;
using bitboard_t = std::uint64_t;
using search_result_t = std::array<int, 2>;

enum TranspositionFlag {
    EXACT = 0,
    LOWER_BOUND = 1,
    UPPER_BOUND = 2
};

struct TranspositionEntry {
    int move;
    int score;
    int depth;
    int flag;
};

#ifdef HISTORY
__gnu_pbds::gp_hash_table<int, int> historyTable[NUM_ROW * NUM_COL]; // History heuristic table
#endif

#ifdef KILLER
static const int MAX_KILLERS = 2;
std::array<std::array<int, MAX_KILLERS>, MAX_DEPTH> killerTable;
#endif

__gnu_pbds::gp_hash_table<std::size_t, TranspositionEntry> transpositionTable;
std::array<std::array<std::array<int, 3>, NUM_COL>, NUM_ROW> zobristTable;
std::mt19937 rng{std::random_device{}()};
std::uniform_int_distribution<int> randDist(0, std::numeric_limits<int>::max());
#ifdef STATICMOVEORDER
static const std::array<int, NUM_COL> staticMovingOrder = {
3,
2,
4,
1,
5,
0,
6,
};
#endif

#define bitboard_get(bitboard, row, col) (((bitboard) >> (((row) << 3) + (col))) & 1)
#define bitboard_set(bitboard, row, col) ((bitboard) |= (1ULL << (((row) << 3) + (col))))
#define bitboard_isfull(bitboard, col) (bitboard_get((bitboard), tobiichi2::NUM_ROW - 1, col) & 1)

void printBoard(const bitboard_t& boardPlayer1, const bitboard_t& boardPlayer2) {
    for (int r = NUM_ROW - 1; r >= 0; r--) {
        for (int c = 0; c < NUM_COL; c++) {
            if (bitboard_get(boardPlayer1, r, c)) {
                cout << player1Disc << " "; // 玩家 1 的棋子
            } else if (bitboard_get(boardPlayer2, r, c)) {
                cout << player2Disc << " "; // 玩家 2 的棋子
            } else {
                cout << ". "; // 空格
            }
        }
        cout << '\n';
    }
}

void printBoard(const board_t& board) {
    for (int r = 0; r < NUM_ROW; r++) {
        for (int c = 0; c < NUM_COL; c++) {
            if (board[r][c] != player1Disc && board[r][c] != player2Disc) {
                cout << ". ";
            } else {
                cout << board[r][c] << " ";
            }
        }
        cout << '\n';
    }
}

std::size_t getZobristHash(const bitboard_t& board1, const bitboard_t& board2) {
    std::size_t hash = 0;
    for (int row = 0; row < NUM_ROW; ++row) {
        for (int col = 0; col < NUM_COL; ++col) {
    // for (int col = 0; col < NUM_COL; ++col) {
    //     for (int row = 0; row < NUM_ROW; ++row) {
            int player = 0;
            if (bitboard_get(board1, row, col)) {
                player = 1;
            } else if (bitboard_get(board2, row, col)) {
                player = 2;
            }
            hash ^= zobristTable[row][col][player];
        }
    }
    return hash;
}

// Generate random values for each board position and disc (2 players)
void initializeZobrist() {
    for (int row = 0; row < NUM_ROW; ++row) {
        for (int col = 0; col < NUM_COL; ++col) {
            zobristTable[row][col][0] = randDist(rng); // Zobrist for EMPTY
            zobristTable[row][col][1] = randDist(rng); // Zobrist for PLAYER1
            zobristTable[row][col][2] = randDist(rng); // Zobrist for PLAYER2
        }
    }
}

inline void makeMove(bitboard_t& board1, bitboard_t& board2, const int col, const char disc) {
    for (int r = 0; r < NUM_ROW; r++) {
        if (!bitboard_get(board1, r, col) && !bitboard_get(board2, r, col)) {
            if (disc == player1Disc) {
                bitboard_set(board1, r, col);
            } else {
                bitboard_set(board2, r, col);
            }
            break;
        }
    }
}

bool checkWinner(const bitboard_t board1, const bitboard_t board2, char disc) {
    bitboard_t bitboard;
    if (disc == player1Disc) {
        bitboard = board1;
    } else if (disc == player2Disc) {
        bitboard = board2;
    } else {
        assert(false && "UNREACHABLE");
    }

    #define rrep(cond) for (int r = 0; r < (cond); r++)
    #define crep(cond) for (int c = 0; c < (cond); c++)

    rrep(NUM_ROW) {
        crep(NUM_COL - 3) {
            if (
                bitboard_get(bitboard, r, c + 0) &
                bitboard_get(bitboard, r, c + 1) &
                bitboard_get(bitboard, r, c + 2) &
                bitboard_get(bitboard, r, c + 3) & 1
            ) {
                return true;
            }
        }
    }

    rrep(NUM_ROW - 3) {
        crep(NUM_COL) {
            if (
                bitboard_get(bitboard, r + 0, c) &
                bitboard_get(bitboard, r + 1, c) &
                bitboard_get(bitboard, r + 2, c) &
                bitboard_get(bitboard, r + 3, c) & 1
            ) {
                return true;
            }
        }
    }

    rrep(NUM_ROW - 3) {
        crep(NUM_COL - 3) {
            if (
                bitboard_get(bitboard, r + 0, c + 0) &
                bitboard_get(bitboard, r + 1, c + 1) &
                bitboard_get(bitboard, r + 2, c + 2) &
                bitboard_get(bitboard, r + 3, c + 3) & 1
            ) {
                return true;
            }
        }
    }

    for (int r = 3; r < NUM_ROW; r++) {
        crep(NUM_COL - 3) {
            if (
                bitboard_get(bitboard, r - 0, c + 0) &
                bitboard_get(bitboard, r - 1, c + 1) &
                bitboard_get(bitboard, r - 2, c + 2) &
                bitboard_get(bitboard, r - 3, c + 3) & 1
            ) {
                return true;
            }
        }
    }

    return false;

    #undef rrep
    #undef crep
}

inline int heuristicFunction(int good, int bad, int empty) {
    int score{};
    if (good == 4) score += 500001;
    else if (good == 3 && empty == 1) score += 5000;
    else if (good == 2 && empty == 2) score += 500;
    else if (bad == 2 && empty == 2) score -= 501;
    else if (bad == 3 && empty == 1) score -= 5001;
    else if (bad == 4) score -= 500000;

    return score;
}

inline int scoreSet(const array<char, 4>& v, const char mydisc, const char oppdisc) {
    int good{}, bad{}, empty{};

    #define ADD(index) do { \
                        good += (v[(index)] == mydisc); \
                        bad += (v[(index)] == mydisc || v[(index)] == oppdisc); \
                        empty += (v[(index)] != mydisc && v[(index)] != oppdisc); \
                      } while (0);

    ADD(0);
    ADD(1);
    ADD(2);
    ADD(3);

    bad -= good;
    return heuristicFunction(good, bad, empty);

    #undef ADD
}

int tabScore(const bitboard_t& board1, const bitboard_t& board2, const char mydisc, const char oppdisc) {
    int score = 0;
    array<char, NUM_COL> rs;
    array<char, NUM_ROW> cs;
    array<char, 4> s;

    for (int r = 0; r < NUM_ROW; r++) {
        for (int c = 0; c < NUM_COL; c++) {
            if (bitboard_get(board1, r, c)) {
                rs[c] = mydisc;
            } else if (bitboard_get(board2, r, c)) {
                rs[c] = oppdisc;
            } else {
                rs[c] = '\0';
            }
        }

        for (int c = 0; c < NUM_COL - 3; c++) {
            std::copy(rs.begin() + c, rs.begin() + c + 4, s.begin());
            score += scoreSet(s, mydisc, oppdisc);
        }
    }

    for (int c = 0; c < NUM_COL; c++) {
        for (int r = 0; r < NUM_ROW; r++) {
            if (bitboard_get(board1, r, c)) {
                cs[r] = mydisc;
            } else if (bitboard_get(board2, r, c)) {
                cs[r] = oppdisc;
            } else {
                cs[r] = '\0';
            }
        }

        for (int r = 0; r < NUM_ROW - 3; r++) {
            std::copy(cs.begin() + r, cs.begin() + r + 4, s.begin());
            score += scoreSet(s, mydisc, oppdisc);
        }
    }

    for (int r = 0; r < NUM_ROW - 3; r++) {
		for (int c = 0; c < NUM_COL - 3; c++) {
			for (int i = 0; i < 4; i++) {
                if (bitboard_get(board1, r + i, c + i)) {
                    s[i] = mydisc;
                } else if (bitboard_get(board2, r + i, c + i)) {
                    s[i] = oppdisc;
                } else {
                    s[i] = '\0';
                }
			}
			score += scoreSet(s, mydisc, oppdisc);
		}
	}
	for (int r = 0; r < NUM_ROW - 3; r++) {
		for (int c = 0; c < NUM_COL - 3; c++) {
			for (int i = 0; i < 4; i++) {
                if (bitboard_get(board1, r + 3 - i, c + i)) {
                    s[i] = mydisc;
                } else if (bitboard_get(board2, r + 3 - i, c + i)) {
                    s[i] = oppdisc;
                } else {
                    s[i] = '\0';
                }
			}
			score += scoreSet(s, mydisc, oppdisc);
		}
	}
	return score;
}

// Move ordering using history and killer heuristics
void orderMoves(vector<int>& moves, int depth) {
#if !defined(HISTORY) && !defined(KILLER)
    UNUSED(moves);
    UNUSED(depth);
#endif

#ifdef HISTORY
    sort(moves.begin(), moves.end(), [&](int a, int b) {
        return historyTable[depth][a] > historyTable[depth][b];
    });
#endif

#ifdef KILLER
    for (int c : moves) {
        if (killerTable[depth][0] == c || killerTable[depth][1] == c) {
            auto it = find(moves.begin(), moves.end(), c);
            if (it != moves.begin()) {
                swap(*it, *moves.begin());
            }
        }
    }
#endif
}

search_result_t miniMax(bitboard_t& board1, bitboard_t& board2, int depth, int alpha, int beta, char currentdisc, const char mydisc, const char oppdisc) {
    std::size_t boardHash = getZobristHash(board1, board2);

    if (transpositionTable.find(boardHash) != transpositionTable.end()) {
        const TranspositionEntry& entry = transpositionTable[boardHash];

        if (entry.depth >= depth) {

# ifdef WTP
            return {entry.score, entry.move};
# endif

            switch (entry.flag) {
                case TranspositionFlag::EXACT:
                    return {entry.score, entry.move};
                case TranspositionFlag::LOWER_BOUND:
                    alpha = std::max(entry.score, alpha);
                    break;
                case TranspositionFlag::UPPER_BOUND:
                    beta = std::min(entry.score, beta);
                    break;
                default:
                    assert(false && "UNREACHABLE");
                    break;
            }

            if (alpha >= beta) {
                return {entry.score, entry.move};
            }
        }
    }

    if (depth <= 0 || checkWinner(board1, board2, mydisc) || checkWinner(board1, board2, oppdisc)) {
        int score = tabScore(board1, board2, mydisc, oppdisc);
        TranspositionEntry entry = {
            .move = -1,
            .score = score,
            .depth = depth,
            .flag = TranspositionFlag::LOWER_BOUND
        };
        transpositionTable[boardHash] = entry;
        return search_result_t{score, -1};
    }

    int bestScore = (currentdisc == mydisc) ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
    int bestMove = -1;

# ifdef NMP
    if (depth > 1 && currentdisc == mydisc) {
        int nullMoveScore = miniMax(board1, board2, depth - 1 - 2, alpha, beta, oppdisc, mydisc, oppdisc)[0];
        if (nullMoveScore >= beta) {
            return {beta, -1};
        }
    }
#endif

    int dynamicDepth = depth;

# ifdef DSE
    if (checkWinner(board1, board2, oppdisc)) {
        dynamicDepth += 1;
    }
# endif

    // Iterate over all possible moves
    int moveCount = 0;

#ifndef LMR
    UNUSED(moveCount);
#endif

    vector<int> moveOrder;
    for (int _c = 0; _c < NUM_COL; _c++) {
        int c = _c;

#ifdef STATICMOVEORDER
        c = staticMovingOrder[_c];
#endif
        if (!bitboard_isfull(board1, c) && !bitboard_isfull(board2, c)) {
            moveOrder.push_back(c);
        }
    }

    orderMoves(moveOrder, depth);

    for (int c : moveOrder) {
        bitboard_t newBoard1 = board1, newBoard2 = board2;
        makeMove(newBoard1, newBoard2, c, currentdisc);

        int currentDepth = dynamicDepth - 1;

#ifdef LMR
        if (moveCount >= NUM_COL / 2 && currentDepth > 3) {
            currentDepth -= 1;
        }
#endif

        int score = miniMax(newBoard1, newBoard2, currentDepth, alpha, beta, (currentdisc == mydisc) ? oppdisc : mydisc, mydisc, oppdisc)[0];

#ifdef HISTORY
        historyTable[depth][c] += (score > bestScore) ? 1 : 0;
#endif

        if ((currentdisc == mydisc && score > bestScore) || (currentdisc != mydisc && score < bestScore)) {
            bestScore = score;
            bestMove = c;

#ifdef KILLER
            if (depth < MAX_DEPTH) {
                if (killerTable[depth][0] != bestMove) {
                    killerTable[depth][1] = killerTable[depth][0];
                    killerTable[depth][0] = bestMove;
                }
            }
#endif
        }

        if (currentdisc == mydisc) {
            alpha = std::max(alpha, bestScore);
        } else {
            beta = std::min(beta, bestScore);
        }

        if (alpha >= beta) break;

        moveCount += 1;
    }

    TranspositionEntry entry{};
    entry.move = bestMove;
    entry.score = bestScore;
    entry.depth = depth;
    entry.flag = TranspositionFlag::EXACT;

    if (bestScore <= alpha) {
        entry.flag = TranspositionFlag::UPPER_BOUND;
    } else if (bestScore >= beta) {
        entry.flag = TranspositionFlag::LOWER_BOUND;
    }

    transpositionTable[boardHash] = entry;
    return {bestScore, bestMove};
}

#ifdef PVS
search_result_t pvs(bitboard_t& board1, bitboard_t& board2, int depth, int alpha, int beta, char currentdisc, const char mydisc, const char oppdisc) {
    std::size_t boardHash = getZobristHash(board1, board2);

    if (transpositionTable.find(boardHash) != transpositionTable.end()) {
        const TranspositionEntry& entry = transpositionTable[boardHash];
        if (entry.depth >= depth) {
#ifdef WTP
            return {entry.score, entry.move};
#endif
            switch (entry.flag) {
                case TranspositionFlag::EXACT:
                    return {entry.score, entry.move};
                case TranspositionFlag::LOWER_BOUND:
                    alpha = std::max(entry.score, alpha);
                    break;
                case TranspositionFlag::UPPER_BOUND:
                    beta = std::min(entry.score, beta);
                    break;
                default:
                    assert(false && "UNREACHABLE");
                    break;
            }

            if (alpha >= beta) {
                return {entry.score, entry.move};
            }
        }
    }

    if (depth <= 0 || checkWinner(board1, board2, mydisc) || checkWinner(board1, board2, oppdisc)) {
        int score = tabScore(board1, board2, mydisc, oppdisc);
        TranspositionEntry entry = {
            .move = -1,
            .score = score,
            .depth = depth,
            .flag = TranspositionFlag::LOWER_BOUND
        };
        transpositionTable[boardHash] = entry;
        return {score, -1};
    }

    int bestScore = (currentdisc == mydisc) ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
    int bestMove = -1;

# ifdef NMP
    if (depth > 1 && currentdisc == mydisc) {
        int nullMoveScore = pvs(board1, board2, depth - 1 - 2, alpha, beta, oppdisc, mydisc, oppdisc)[0];
        if (nullMoveScore >= beta) {
            return {beta, -1};
        }
    }
#endif

    int dynamicDepth = depth;
#ifdef DSE
    if (checkWinner(board1, board2, oppdisc)) {
        dynamicDepth += 1;
    }
#endif

    vector<int> moveOrder;
    for (int c = 0; c < NUM_COL; c++) {
        if (!bitboard_isfull(board1, c) && !bitboard_isfull(board2, c)) {
            moveOrder.push_back(c);
        }
    }

    orderMoves(moveOrder, depth);

    bool isFirstMove = true;

    for (int c : moveOrder) {
        bitboard_t newBoard1 = board1, newBoard2 = board2;
        makeMove(newBoard1, newBoard2, c, currentdisc);

        int score;

        int currentDepth = dynamicDepth - 1;

#ifdef LMR
        if (moveCount >= NUM_COL / 2 && currentDepth > 3) {
            currentDepth -= 1;
        }
#endif

        if (isFirstMove) {
            score = pvs(newBoard1, newBoard2, currentDepth, alpha, beta, (currentdisc == mydisc) ? oppdisc : mydisc, mydisc, oppdisc)[0];
            isFirstMove = false;
        } else {
            score = pvs(newBoard1, newBoard2, currentDepth, alpha - 1, beta + 1, (currentdisc == mydisc) ? oppdisc : mydisc, mydisc, oppdisc)[0];
            if (beta > score && score < alpha) {
                score = pvs(newBoard1, newBoard2, currentDepth, alpha, beta, (currentdisc == mydisc) ? oppdisc : mydisc, mydisc, oppdisc)[0];
            }
        }

#ifdef HISTORY
        historyTable[depth][c] += (score > bestScore) ? 1 : 0;
#endif

        if ((currentdisc == mydisc && score > bestScore) || (currentdisc != mydisc && score < bestScore)) {
            bestScore = score;
            bestMove = c;

#ifdef KILLER
            if (depth < MAX_DEPTH) {
                if (killerTable[depth][0] != bestMove) {
                    killerTable[depth][1] = killerTable[depth][0];
                    killerTable[depth][0] = bestMove;
                }
            }
#endif
        }

        if (currentdisc == mydisc) {
            alpha = std::max(alpha, bestScore);
        } else {
            beta = std::min(beta, bestScore);
        }

        if (alpha >= beta) break;
    }

    TranspositionEntry entry = {
        .move = bestMove,
        .score = bestScore,
        .depth = depth,
        .flag = TranspositionFlag::EXACT,
    };

    if (bestScore <= alpha) {
        entry.flag = TranspositionFlag::UPPER_BOUND;
    } else if (bestScore >= beta) {
        entry.flag = TranspositionFlag::LOWER_BOUND;
    }

    transpositionTable[boardHash] = entry;
    return {bestScore, bestMove};
}
#endif

#if defined(MTDF_MINIMAX) || defined(MTDF_PVS)
search_result_t mtdf(bitboard_t& board1, bitboard_t& board2, int depth, int f, const char mydisc, const char oppdisc) {
    int top = std::numeric_limits<int>::max();
    int down = std::numeric_limits<int>::min();
    int g = f;
    int beta;
    int bestMove;

    while (down < top) {
        if (g == down) {
            beta = g + 1;
        } else {
            beta = g;
        }

        auto result = search_result_t{};
#if defined(MTDF_MINIMAX)
        result = miniMax(board1, board2, depth, beta - 1, beta, mydisc, mydisc, oppdisc);
#elif defined(MTDF_PVS)
        result = pvs(board1, board2, depth, beta - 1, beta, mydisc, mydisc, oppdisc);
#endif

        g = result[0];
        if (g < beta) {
            top = g;
        } else {
            down = g;
        }

        bestMove = result[1];
    }

    return {g, bestMove};
}
#endif

#ifdef MTDF_IDDFS
search_result_t IDDFS(bitboard_t& board1, bitboard_t& board2, int maxDepth,  const char mydisc, const char oppdisc) {
    int f = tabScore(board1, board2, mydisc, oppdisc);
    int bestMove = -1;
    clock_t t0 = clock();
    clock_t t1;
    for (int depth = 1; depth <= maxDepth; depth++) {
        auto result = mtdf(board1, board2, depth, f, mydisc, oppdisc);
        t1 = clock() - t0;
        f = result[0];
        bestMove = result[1];
        if ((double)t1 / CLOCKS_PER_SEC > 0.5) {
            break;
        }
    }

    return {f, bestMove};
}
#endif

} // namespace tobiichi2

namespace E2413OOOO
// ===================================================
{
    tobiichi2::bitboard_t bitboard1, bitboard2;
    bool isNewBoard(const tobiichi2::board_t& board, char mydisc, char oppdisc) {
        int cnt = 0;
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 7; c++) {
                cnt += board[r][c] == mydisc || board[r][c] == oppdisc;
                if (cnt > 1) goto end;
            }
        }
end:
        return cnt <= 1;
    }

    int getMove(vector<vector<char>> board, char mydisc, char oppdisc)
    {
        if (!tobiichi2::gIsZobristSeted) {
            tobiichi2::initializeZobrist();
            tobiichi2::gIsZobristSeted = true;
        }

        if (isNewBoard(board, mydisc, oppdisc)) {
#ifndef CLEARTP
            tobiichi2::transpositionTable.clear();
#endif
            bitboard1 = 0ULL;
            bitboard2 = 0ULL;
            tobiichi2::player1Disc = mydisc;
            tobiichi2::player2Disc = oppdisc;

#ifdef KILLER
            for (int i = 0; i < tobiichi2::MAX_DEPTH; i++) {
                tobiichi2::killerTable[i][0] = tobiichi2::killerTable[i][1] = -3227;
            }
#endif

#ifdef HISTORY
            for (auto &i : tobiichi2::historyTable) {
                i.clear();
            }
#endif
        }

        for (int r = tobiichi2::NUM_ROW - 1; r >= 0; r--) {
            for (int c = 0; c < 7; c++) {
                if (board[r][c] == oppdisc && !bitboard_get(bitboard2, tobiichi2::NUM_ROW - r - 1, c)) {
                    bitboard_set(bitboard2, tobiichi2::NUM_ROW - r - 1, c);
                    break;
                }
            }
        }

        int col = -1;

#if defined(MINIMAX)
        col = tobiichi2::miniMax(bitboard1, bitboard2, tobiichi2::MAX_DEPTH, numeric_limits<int>::min(), numeric_limits<int>::max(), mydisc, mydisc, oppdisc)[1];
#elif defined(PVS)
        col = tobiichi2::pvs(bitboard1, bitboard2, tobiichi2::MAX_DEPTH, numeric_limits<int>::min(), numeric_limits<int>::max(), mydisc, mydisc, oppdisc)[1];
#elif defined(MTDF_IDDFS) && (defined(MTDF_MINIMAX) || defined(MTDF_PVS))
        col = tobiichi2::IDDFS(bitboard1, bitboard2, tobiichi2::MAX_DEPTH, mydisc, oppdisc)[1];
#elif defined(MTDF_MINIMAX) || defined(MTDF_PVS)
        col = tobiichi2::mtdf(bitboard1, bitboard2, tobiichi2::MAX_DEPTH, 0, mydisc, oppdisc)[1];
        if (col == -1) {
            col = tobiichi2::miniMax(bitboard1, bitboard2, tobiichi2::MAX_DEPTH, numeric_limits<int>::min(), numeric_limits<int>::max(), mydisc, mydisc, oppdisc)[1];
        }
#endif

        if (col == -1) {
            cout << "col = -1" << '\n';
            // tobiichi2::printBoard(bitboard1, bitboard2);
            cout << mydisc << " " << oppdisc << '\n';

            for (int c = 0; c < tobiichi2::NUM_COL; c++) {
                if (!bitboard_isfull(bitboard1, c) && !bitboard_isfull(bitboard2, c)) {
                    col = c;
                    break;
                }
            }
        }
        tobiichi2::makeMove(bitboard1, bitboard2, col, mydisc);
        return col;
    }

} // namespace E2413OOOO
