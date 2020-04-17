import pygame
import numpy as np

class Pong():
    def __init__(self):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.screensize = (800, 600)
        self.top_border = 20
        self.ball_size = 15

        # Starting coordinates of the paddle
        self.paddle_x = 400
        self.paddle_y = 580

        # initial speed of the paddle
        self.paddle_change_x = 0
        self.paddle_change_y = 0

        # initial position of the ball, randomly changed
        self.ball_x = np.random.randint(10, 700)
        self.ball_x -= self.ball_x % 5
        self.ball_y = np.random.randint(30, 100)
        self.ball_y -= self.ball_y % 5


        # speed of the ball
        self.ball_change_x = 5
        self.ball_change_y = 5

        self.score = 0
        self.done = False

        # For running pygame.init() only once
        self.render_started = False

        # ripped from gym package
        """
        Actions:
            Type: Discrete(2)
            Num	Action
            0	Push paddle to the left
            1   No movement
            2	Push paddle to the right

        Observation: 
            Type: Box(3)
            Num	Observation         Min         Max
            0	Ball x               0          800
            1	Ball y              20          600
            2	Paddle x             0          700
        """
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0, 20, 0], dtype=int), high=np.array([800, 600, 700], dtype=int),
                                     dtype=np.int)

    # draws the paddle. Also restricts its movement between the edges of the window.
    def drawrect(self, screen, x, y):
        """ Drawing the paddle"""
        if x <= 0:
            x = 0
        if x >= 699:
            x = 699

        if self.render_started:
            pygame.draw.rect(self.screen, self.RED, [x, y, 100, 20])

    def reset(self):
        """ Reset the game """

        # Starting coordinates of the paddle
        self.paddle_x = 400  # np.random.randint(200, 500)
        self.paddle_y = 580

        # initial speed of the paddle
        self.paddle_change_x = 0
        self.paddle_change_y = 0

        # initial position of the ball
        self.ball_x = np.random.randint(10, 700)
        self.ball_x -= self.ball_x % 5
        self.ball_y = np.random.randint(30, 100)
        self.ball_y -= self.ball_y % 5

        # speed of the ball
        self.ball_change_x = 5
        self.ball_change_y = 5

        self.score = 0
        self.done = False

        self.clock = pygame.time.Clock()
        state = [self.ball_x, self.ball_y, self.paddle_x]
        return np.array(state)

    def processevent(self):
        """ Process keyboard and mouse events """
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                pass

            if event.type == pygame.QUIT:           # Terminates the game
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()

    def render(self, mode):
        """ Show the game GUI. It need to be off for faster training of the agent """

        # used by keras-rl to visualize the game
        if mode == 'human':
            if self.render_started == False:
                pygame.init()

                # Initializing the display window
                self.screen = pygame.display.set_mode(self.screensize)
                pygame.display.set_caption("Pong")

                self.render_started = True

            # process keyboard and mouse events
            self.processevent()

            # Update the score board
            font = pygame.font.SysFont('Calibri', 15, False, False)
            text = font.render("Score = " + str(self.score), True, self.WHITE)
            self.screen.blit(text, [400, 5])

            pygame.draw.line(self.screen, self.WHITE, (0, 20), (800, 20), 1)
            pygame.display.flip()
            self.clock.tick(300)

    # game's main loop
    def step(self, action):
        """ Main loop of the Pong game"""

        # reward for each action
        reward = 1

        if action == 0:                     # left move
            self.paddle_change_x = -6
        elif action == 1:                   # no move
            self.paddle_change_x = 0
        else:                               # right move
            self.paddle_change_x = 6

        # change the position of the paddle, based on the movement
        self.paddle_x += self.paddle_change_x
        self.paddle_y += self.paddle_change_y

        # change the ball position
        self.ball_x += self.ball_change_x
        self.ball_y += self.ball_change_y

        # handling the movement of the ball.
        if self.ball_x <= 0:
            self.ball_x = 0
            self.ball_change_x *= -1
        elif self.ball_x >= 785:
            self.ball_x = 785
            self.ball_change_x *= -1
        elif self.ball_y <= self.top_border:
            self.ball_y = self.top_border
            self.ball_change_y *= -1
        elif self.paddle_x <= 0:
            self.paddle_x = 0
        elif self.paddle_x >= 699:
            self.paddle_x = 699

        # reward policy
        if self.paddle_x <= self.ball_x <= self.paddle_x + 100 and (self.ball_y == 565 and self.ball_change_y > 0):
            self.ball_change_y *= -1
            reward += 10
            self.score += 1
        elif self.ball_y >= 585:
            self.ball_y = 585
            self.ball_change_y *= -1
            reward = 0
            self.done = True

        # draw the ball and paddle of the GUI has been created
        if self.render_started:
            self.screen.fill(self.BLACK)
            # draw ball
            pygame.draw.rect(self.screen, self.WHITE, [self.ball_x, self.ball_y, self.ball_size, self.ball_size])

            # draw paddle
            self.drawrect(self.screen, self.paddle_x, self.paddle_y)

        # return the current state
        state = [self.ball_x, self.ball_y, self.paddle_x]
        return state, reward, self.done, {}


# OpenAI gym classes for initializing the game environment
class Space(object):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """
    def __init__(self, shape=None, dtype=None):
        import numpy as np  # takes about 300-400ms to import, so we load lazily
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self.np_random = None
        self.seed()

    def sample(self):
        """Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        #self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n


class Discrete(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.

    Example::

        >>> Discrete(2)

    """
    def __init__(self, n):
        assert n >= 0
        self.n = n
        super(Discrete, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n


class Box(Space):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)

    """

    def __init__(self, low, high, shape=None, dtype=np.float32):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)

        if shape is None:
            assert low.shape == high.shape, 'box dimension mismatch. '
            self.shape = low.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high), 'box requires scalar bounds. '
            self.shape = tuple(shape)
            self.low = np.full(self.shape, low)
            self.high = np.full(self.shape, high)

        def _get_precision(dtype):
            if np.issubdtype(dtype, np.floating):
                return np.finfo(dtype).precision
            else:
                return np.inf

        low_precision = _get_precision(self.low.dtype)
        high_precision = _get_precision(self.high.dtype)
        dtype_precision = _get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            #logger.warn("Box bound precision lowered by casting to {}".format(self.dtype))
            pass
        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high

        super(Box, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = np.all(self.bounded_below)
        above = np.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high if self.dtype.kind == 'f' \
            else self.high.astype('int64') + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(
            size=unbounded[unbounded].shape)

        sample[low_bounded] = self.np_random.exponential(
            size=low_bounded[low_bounded].shape) + self.low[low_bounded]

        sample[upp_bounded] = -self.np_random.exponential(
            size=upp_bounded[upp_bounded].shape) + self.high[upp_bounded]

        sample[bounded] = self.np_random.uniform(low=self.low[bounded],
                                                 high=high[bounded],
                                                 size=bounded[bounded].shape)
        if self.dtype.kind == 'i':
            sample = np.floor(sample)

        return sample.astype(self.dtype)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, Box) and (self.shape == other.shape) and np.allclose(self.low,
                                                                                      other.low) and np.allclose(
            self.high, other.high)
