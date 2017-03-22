# 
# Project Eagle Eye
# Group 15 - UniSA 2015
# 
# Gwilyn Saunders
# version 0.1.1
# 

class Theta:
    NonDual    = 0
    Both       = 0
    Backside   = 1
    Left       = 1
    Buttonside = 2
    Right      = 2
    
    @staticmethod
    def side(var):
        return 'Both' if var == 0 else 'Left' if var == 1 else 'Right'
    
    @staticmethod
    def name(var):
        return 'NonDual' if var == 0 else 'Backside' if var == 1 or var == 'backside' else 'Buttonside' #added lowercase backside. ~Will.
    
    @staticmethod
    def resolve(var):
        var = var.lower()
        if 'back' in var \
                or 'left' in var:
            return Theta.Left
        elif 'button' in var \
                or 'right' in var:
            return Theta.Right
        else:
            return Theta.Both
