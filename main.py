import utility
from config import *

if __name__ == '__main__':
    field = utility.field(teamA_num, teamB_num, field_width, field_length)
    field.match(20)
    # field.test_collide(20)