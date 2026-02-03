import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pickle
import os
from Transformerv2 import ReversiBotDecoder
import anytree as anytree

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
        def __init__(self, board = None, move_str = '', player = -1):
            if board is not None:
                self.board = board.copy()
            else:
                self.board = self._init_board()
            self.player = player
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
            self.player = self.player*(-1)
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


def chunk_string(s):
    """Split string into 2-character chunks"""
    return [s[i:i+2] for i in range(0, len(s), 2)]


class ReversAI:
        
    def __init__(self,model_pth,model_player,token_to_hot,idx_to_token,token_to_idx):
            self.game = ReversiGame()
            self.model = self.load_model(model_pth)
            self.model_player = model_player
            self.move_str = ''

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

    def get_best_line(self, depth, start_game, idx_to_token):
        step = 0
        root = tree_node(parent=None,game=start_game,children=None, value=None, value_str=start_game.move_str)
        depth_tree = [[] for _ in range(depth+1)]
        depth_tree[0].append(root)
        current_player = self.model_player

        for step in range(depth):
            for node in depth_tree[step]:
                temp_board = node.game.board.copy()
                temp_str = node.game.move_str
                #Calculating legal moves for for depth
                for move in node.game.get_legal_moves(current_player):
                    
                    
                    temp_game = ReversiGame(temp_board,temp_str,current_player)
                    temp_game.make_move(move[0],move[1],current_player,idx_to_token)
                    
                    evaluation = self.evaluate_position(temp_game.move_str)

                    value = evaluation['probabilities']
                    leaf = depth_tree[step+1].append(tree_node(parent=node,game = temp_game, value=value, value_str=temp_game.move_str))
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
                'black': probabilities[0, 0].item(),
                'white': probabilities[0, 1].item(),
                'draw': probabilities[0, 2].item(),
            }
        }

    def minimax(node, depth, is_maximizing):
        if depth == 0 or node.is_terminal():
            return move_str
        
        if is_maximizing:
            best_value = 0
            for child in node.children:
                value = minimax(child,depth-1, False)
                best_value = max(best_value, child.value['probabilities'[{-1:'black', 1: 'white'}[self.model_player]]])
            return best_value
        else:
            best_value = 1
            for child in node.children:
                value = minimax(child, depth-1, True)
                best_value = min(best_value, child.value['probabilities'[{-1:'black', 1: 'white'}[-1*self.model_player]]])
            return best_value
        

    def make_move(self, move):
        self.game.make_move(move)
        self.move_str = self.move_str.cat(move)
        

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
        for child in self.children:
            if child is None:
                return True
            else: 
                return False

    def print(self):
        print(self.value_str)
        for child in self.children:
            child.print()            


def print_prob_tree(node):
    if node.is_terminal:
        return ''
    else:
        for child in node.children:
            print_prob_tree(child)
            print(child.value)

def Testing():
    model_pth = os.path.join('ReversAI','Success_V2_Model','ReproV2FS.pth')
    bot = ReversAI(model_pth, -1, token_to_hot, idx_to_token, token_to_idx)
    root,depth_tree = bot.get_best_line(4,bot.game,idx_to_token)
    print_prob_tree(root)
    root.validate_integrity()
    

Testing()









    
