import numpy as np
import torch
import os
from Transformerv2 import ReversiBotDecoder

###CONSTANTS
     # Create token mappings - CORRECTED
letters = 'abcdefgh'  # columns: a=0, b=1, ..., h=7
numbers = '87654321'  # rows: 8=0 (top), 7=1, ..., 1=7 (bottom)
token_to_idx = {}
for row_idx, number in enumerate(numbers):
    for col_idx, letter in enumerate(letters):
        token = letter + number
        token_to_idx[token] = (row_idx, col_idx)  # (row, col)
    
idx_to_token = {}
for row_idx, number in enumerate(numbers):
    for col_idx, letter in enumerate(letters):
        token = letter + number
        idx = (row_idx, col_idx)
        idx_to_token[idx] =  token # (row, col)

numbers = '12345678'
token_to_hot = {}
i = 0
for letter in letters:
    for number in numbers:
        token = letter+number
        token_to_hot[token] = i 
        i = i+1
    

class ReversiGame:
        def __init__(self, board = None, move_str = ''):
            if board is not None:
                self.board = board.copy()
            else:
                self.board = self._init_board()
            self.move_str = move_str
        def _init_board(self):
            board = np.zeros((8, 8), dtype=int)
            board[3, 4] = 1   # white
            board[4, 3] = 1   # white
            board[3, 3] = -1  # black
            board[4, 4] = -1  # black
            return board
        
        def get_legal_moves(self, player):

            legal_moves = []
            
            for row in range(8):
                for col in range(8):
                    if self.is_legal_move(row, col, player):
                        legal_moves.append((row, col))
            
            return legal_moves
        
        def is_legal_move(self, row, col, player):
            """Check if a move is legal"""
            if self.board[row, col] != 0:
                return False
            
            # Check all 8 directions
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                        (0, 1), (1, -1), (1, 0), (1, 1)]
            
            for dr, dc in directions:
                if self._has_flips(row, col, player, dr, dc):
                    return True
            
            return False
        
        def _has_flips(self, row, col, player, dr, dc):
            r, c = row + dr, col + dc
            opponent = -player
            found_opponent = False
            
            while 0 <= r < 8 and 0 <= c < 8:
                if self.board[r, c] == opponent:
                    found_opponent = True
                elif self.board[r, c] == player:
                    return found_opponent
                else:
                    return False
                r += dr
                c += dc
            
            return False
        
        def make_move(self, row, col, player, idx_to_token):
            if not self.is_legal_move(row, col, player):
                raise ValueError(f"Illegal move: at ({row}, {col})")
            self.move_str = self.move_str+idx_to_token[(row,col)]
            self.board[row, col] = player
            
            # Flip opponent pieces in all directions
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                        (0, 1), (1, -1), (1, 0), (1, 1)]
            
            for dr, dc in directions:
                self._flip_pieces(row, col, player, dr, dc)
            return (row, col)
        def set_board(board):
            self.board = board

        def _flip_pieces(self, row, col, player, dr, dc):
            r, c = row + dr, col + dc
            opponent = -player
            pieces_to_flip = []
            
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == opponent:
                pieces_to_flip.append((r, c))
                r += dr
                c += dc

            if 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                for flip_r, flip_c in pieces_to_flip:
                    self.board[flip_r, flip_c] = player
        def printBoard(self):
            print(self.board)

        def is_terminal(self):
            return not self.get_legal_moves(1) and not self.get_legal_moves(-1)

        def get_score(self):
            return np.sum(self.board)

        def get_winner(self):
            if self.is_terminal():
                score = self.get_score()
                if score > 0:
                    return 1
                elif score < 0:
                    return -1
                else:
                    return 0
            return None

def chunk_string(s):
    """Split string into 2-character chunks"""
    return [s[i:i+2] for i in range(0, len(s), 2)]


class ReversAI:
        
    def __init__(self,model_pth = None, model_player = None, depth=4, token_to_hot = None, idx_to_token = None, token_to_idx = None):
            self.game = ReversiGame()
            self.model = self.load_model(model_pth)
            self.model_player = model_player
            self.move_str = ''
            self.depth = depth

            self.token_to_hot = token_to_hot
            self.idx_to_token = idx_to_token
            self.token_to_idx = token_to_idx

    def load_model(self,model_pth, device='cuda'):
        """Load the saved model from disk"""
        # Initialize model with the same hyperparameters used in training
        model = ReversiBotDecoder(
            vocab_size=64,
            embed_size=256,
            num_layers=5,
            heads=4,
            dropout=0.1,
            device=device,
            max_length=60,
            forward_expansion=3,
            num_classes=3
            )
        
        # Load the state dict
        state_dict = torch.load(model_pth)
        
        # Load weights into model
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        return model.to(device)

    """
    Args: max_depth, moves_strs from tree
    Return: the highest quality moves based on leaves from the possible moves at depth 4
    This function receives a list of 
    """

    def get_best_line(self):
        depth = self.depth
        root = tree_node(parent=None,game = ReversiGame(self.game.board.copy(), move_str=self.game.move_str),children=None, value=None, value_str=self.game.move_str)
        depth_tree = [[] for _ in range(depth+1)]
        depth_tree[0].append(root)
        current_player = self.model_player
        

        for step in range(depth):
            for node in depth_tree[step]:
                temp_board = node.game.board.copy()
                temp_str = node.game.move_str
                #Calculating legal moves for for depth
                for move in node.game.get_legal_moves(current_player):
                    row, col = move
                    temp_game = ReversiGame(temp_board,temp_str)
                    temp_game.make_move(row = row, col = col, player = current_player ,idx_to_token = self.idx_to_token)
                    
                    evaluation = self.evaluate_position(temp_game.move_str)

                    value = evaluation['probabilities']
                    leaf = tree_node(parent=node,game = temp_game, value=value, value_str=temp_game.move_str)
                    depth_tree[step+1].append(leaf)
                    node.append_child(leaf)
            current_player = -current_player
        return root, depth_tree

    def evaluate_position(self,move_str):
        self.model.eval()
        moves = chunk_string(move_str)
        move_count = len(moves)
       
        encoded_moves = torch.tensor(
            [self.token_to_hot[move] for move in moves],
            dtype=torch.long
        ).unsqueeze(0).to(self.model.device)
        
        turns = torch.tensor(
            [(-1) ** (i + 1) for i in range(move_count)],
            dtype=torch.long
        ).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            logits = self.model(encoded_moves, turns=turns)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        outcome_map = {0: 'black', 1: 'white', 2: 'draw'}
    
        return {
            'prediction': outcome_map[prediction],
            'prediction_id': prediction,
            'probabilities': {
                '-1': probabilities[0, 0].item(),
                '1': probabilities[0, 1].item(),
                '0': probabilities[0, 2].item(),
            }
        }

    def minimax(self, node, depth, is_maximizing):
        if depth == 0 or node.is_terminal():
            moves_in_path = chunk_string(node.value_str.replace(self.game.move_str,""))
            return node.value[str(self.model_player)],node, moves_in_path
        
        if is_maximizing:
            best_value = -float('inf')
            best_node = None
            for child in node.children:
                value, child_node, path = self.minimax(child, depth-1, False)
                if value > best_value:
                    best_value = value
                    best_path = path
                    best_node = child
            return best_value, best_node, best_path
        else:
            best_value = float('inf')
            best_node = None
            for child in node.children:
                value, child_node, path = self.minimax(child, depth-1, True)
                if value < best_value:
                    best_value = value
                    best_node = child
                    best_path = path
            return best_value, best_node, best_path 

    def get_best_move(self):
        root, depth_tree = self.get_best_line()
        best_value, best_node, best_path = self.minimax(root, self.depth, True)
        return best_path[0]

    def make_move(self, idx_move, player):
        row, col = idx_move
        move = self.game.make_move(row,col,player,self.idx_to_token)

        self.move_str = self.move_str+self.idx_to_token[move]
        return move

    def play(self):
        self.game.printBoard()
        if not self.game.get_legal_moves(self.model_player):
            print("No legal moves available", self.game.get_legal_moves(self.model_player))
            return None
        if not self.game.is_terminal():
            token_move = self.get_best_move()  
            idx_move = self.make_move(token_to_idx[token_move], player = self.model_player)
            return idx_move
        else:
            print("Game Over")
            return {self.model_player:'win',-self.model_player:'loss',0:'draw'}[self.game.get_score()]



class tree_node:
    def __init__(self, parent=None, game = None,children=None, value = None, value_str=None):
        self.parent = parent
        self.game = game
        self.children = children if children is not None else []
        self.value = value
        self.value_str = value_str
    
    def get_parent(self):
        return self.parent
    def get_children(self):
        return self.children
    def get_value(self):
        return self.value
    def get_value_str(self):
        return self.value_str

    def set_parent(self,parent):
        self.parent = parent
    def set_children(self,children):
        self.children = children
    
    def append_child(self,child):
        self.children.append(child)
    
    def validate_integrity(self):

        # Check if node has no children (leaf node)
        if not self.children:
            return True
        
        # Validate all children
        for child in self.children:
            # Check that parent's value_str is in child's value_str
            if child is None:
                print(f"Warning: Node '{self.value_str}' has none child")
            else:
                if self.value_str and child.value_str:
                    if self.value_str not in child.value_str:
                        print(f"Integrity Error: '{self.value_str}' not found in child '{child.value_str}'")
                        return False
                
                # Recursively validate child's subtree
                if not child.validate_integrity():
                    return False
            
        return True
    def is_terminal(self): 
        return not self.children

    def print(self):
        print(self.value_str)
        for child in self.children:
            child.print()            


def matchConductor(bot1, bot2):
    if bot1.model_player == bot2.model_player:
        raise ValueError("Bot1 and Bot2 must have different model players")
    if bot1.model_player == 1:
        white_bot = bot1
        black_bot = bot2
    else:
        white_bot = bot2
        black_bot = bot1
    game_history = []
    while not white_bot.game.is_terminal() and not black_bot.game.is_terminal():
        
        print("black bots legal moves: ",black_bot.game.get_legal_moves(-1))
        idx_move = black_bot.play()
        if idx_move is None:
            print("No legal moves for black")
        else:
            row, col = idx_move
            white_bot.game.make_move(row = row, col=col,player= -1, idx_to_token=idx_to_token)
            game_history.append(idx_to_token[idx_move])

        print("white bots legal moves: ",white_bot.game.get_legal_moves(1))
        idx_move = white_bot.play()
        if idx_move is None:
            print("No legal moves for white")
        else:
            row, col = idx_move
            black_bot.game.make_move(row = row, col=col,player = 1, idx_to_token=idx_to_token)
            game_history.append(idx_to_token[idx_move])

        print("Game Score = ", white_bot.game.get_score())
        print("Game History: ", game_history)
    print("Game Over : ", white_bot.game.get_winner())
    return white_bot.game.get_winner(), game_history




def PlayTest():
    model_pth = os.path.join('Success_V2_Model','ReproV2FS.pth')
    

    
    greedy_black_bot = ReversAI(
        model_pth = model_pth,
        model_player=-1,
        depth=1, 
        token_to_hot = token_to_hot,
        idx_to_token = idx_to_token, 
        token_to_idx = token_to_idx
         )
    

    deep_black_bot = ReversAI(
        model_pth = model_pth,
        model_player=-1,
        depth=4, 
        token_to_hot = token_to_hot,
        idx_to_token = idx_to_token, 
        token_to_idx = token_to_idx
         )
    super_deep_black_bot = ReversAI(
        model_pth = model_pth,
        model_player=-1,
        depth=8, 
        token_to_hot = token_to_hot,
        idx_to_token = idx_to_token, 
        token_to_idx = token_to_idx
         )

    greedy_white_bot = ReversAI(
        model_pth = model_pth,
        model_player=1,
        depth=1, 
        token_to_hot = token_to_hot,
        idx_to_token = idx_to_token, 
        token_to_idx = token_to_idx
         )

    
    deep_white_bot = ReversAI(
        model_pth = model_pth, 
        model_player = 1,
        depth=4, 
        token_to_hot = token_to_hot,
        idx_to_token = idx_to_token, 
        token_to_idx = token_to_idx
        )
    all_games = []
    
    score, history = matchConductor(super_deep_black_bot, greedy_white_bot)
    all_games.append((score, history))
    score, history = matchConductor(deep_black_bot, deep_white_bot)
    all_games.append((score, history))
    score, history = matchConductor(greedy_black_bot, greedy_white_bot)
    all_games.append((score, history))
    score, history = matchConductor(greedy_black_bot, deep_white_bot)
    all_games.append((score, history))
        

PlayTest()









    
