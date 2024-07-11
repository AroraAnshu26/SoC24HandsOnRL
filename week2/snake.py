import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 180, 180)
RED = (255, 0, 150)
BLACK = (0, 0, 0)

# Snake settings
SNAKE_SIZE = 20
SNAKE_SPEED = 10

# Food settings
FOOD_SIZE = 20

# Initialize snake
snake = [[100, 60], [90, 60], [80, 60]]   # this is just the length of the snake 
snake_direction = 'RIGHT'
change_to = snake_direction

# Initialize food
food_pos = [random.randrange(1, (WIDTH // FOOD_SIZE)) * FOOD_SIZE,
            random.randrange(1, (HEIGHT // FOOD_SIZE)) * FOOD_SIZE]
food_spawn = True

# Initialize score
score = 0
font = pygame.font.SysFont('arial', 35)

# Game Over
def game_over():
    global score

    # Display final score
    SCREEN.fill(WHITE)
    game_over_surface = font.render(f'Game Over! Your Score: {score}', True, RED)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (WIDTH / 2, HEIGHT / 4)
    SCREEN.blit(game_over_surface, game_over_rect)

    # Display options
    restart_surface = font.render('Press R to Restart or Q to Quit', True, BLACK)
    restart_rect = restart_surface.get_rect()
    restart_rect.midtop = (WIDTH / 2, HEIGHT / 2)
    SCREEN.blit(restart_surface, restart_rect)

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_r:
                    main()

# Display score
def show_score():
    score_surface = font.render('Score: ' + str(score), True, BLACK)
    score_rect = score_surface.get_rect()
    score_rect.midtop = (WIDTH / 2, 10)
    SCREEN.blit(score_surface, score_rect)

# Main Function
def main():
    global change_to, snake_direction, food_spawn, food_pos, score, snake

    # Reset the game variables
    snake = [[100, 60], [90, 60], [80, 60]]
    change_to = snake_direction
    food_pos = [random.randrange(1, (WIDTH // FOOD_SIZE)) * FOOD_SIZE,
                random.randrange(1, (HEIGHT // FOOD_SIZE)) * FOOD_SIZE]
    food_spawn = True
    score = 0
    
    game_on = True
    while game_on:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_on = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    if snake_direction != 'DOWN':
                        change_to = 'UP'
                if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    if snake_direction != 'UP':
                        change_to = 'DOWN'
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    if snake_direction != 'RIGHT':
                        change_to = 'LEFT'
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    if snake_direction != 'LEFT':
                        change_to = 'RIGHT'

        # Validate direction
        if change_to == 'UP' and snake_direction != 'DOWN':
            snake_direction = change_to
        if change_to == 'DOWN' and snake_direction != 'UP':
            snake_direction = change_to
        if change_to == 'LEFT' and snake_direction != 'RIGHT':
            snake_direction = change_to
        if change_to == 'RIGHT' and snake_direction != 'LEFT':
            snake_direction = change_to

        # Update the position of the snake
        if snake_direction == 'UP':
            snake[0][1] -= SNAKE_SIZE
        if snake_direction == 'DOWN':
            snake[0][1] += SNAKE_SIZE
        if snake_direction == 'LEFT':
            snake[0][0] -= SNAKE_SIZE
        if snake_direction == 'RIGHT':
            snake[0][0] += SNAKE_SIZE

        # Growing mechanism
        # i would rather call it as moving mechanism
        snake.insert(0, list(snake[0]))   #pehle same waalacopy kara, fir agle iteration main it gets changed
        if snake[0] == food_pos:
            food_spawn = False
            score += 10 
            snake.insert(0, list(snake[0]))
            print("Food eaten! Score:", score)
        else:
           snake.pop()                       #moving ahead ke liye last waala pop kar diya

        if not food_spawn:
            food_pos = [random.randrange(1, (WIDTH // FOOD_SIZE)) * FOOD_SIZE,
                        random.randrange(1, (HEIGHT // FOOD_SIZE)) * FOOD_SIZE]
            print("New food position:", food_pos)
        food_spawn = True

        # Background
        SCREEN.fill(WHITE)

        # Draw snake, each of the element of list-block in the snake represents a green box in it, here the boxes are overlapping
        for pos in snake:
            if (pos != snake[0]):
                pygame.draw.rect(SCREEN, GREEN, pygame.Rect(pos[0], pos[1], SNAKE_SIZE, SNAKE_SIZE))
            else:
                pygame.draw.rect(SCREEN, BLACK, pygame.Rect(pos[0], pos[1], SNAKE_SIZE, SNAKE_SIZE))
         #Drawing the food
        pygame.draw.rect(SCREEN, RED, pygame.Rect(food_pos[0], food_pos[1], FOOD_SIZE, FOOD_SIZE))

        # Check for collisions
        if (snake[0][0] < 0 or snake[0][0] >= WIDTH or
                snake[0][1] < 0 or snake[0][1] >= HEIGHT):
            print("Snake hit the boundary!")
            game_over()
        for block in snake[3:]:    #it can never hit itself earlier
            if snake[0] == block:
                print("Snake hit itself!")
                #print(food_pos)
                #print(snake[0])
                #print(snake)
                game_over()

        # Display score
        show_score()

        # Refresh game screen
        pygame.display.flip()

        # Frame Per Second /Refresh Rate
        CLOCK.tick(SNAKE_SPEED)

    pygame.quit()
    quit()
main()
# Run the game
'''try:
    main()
except Exception as e:
    print("An error occurred:", e)
    pygame.quit()
    quit()'''
