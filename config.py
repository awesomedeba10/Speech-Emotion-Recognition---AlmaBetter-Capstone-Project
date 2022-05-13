APP_ENV = 'Development'
APP_HOST = '127.0.0.1'
APP_PORT = 8080
APP_DEBUG = True

# Define the application directory
import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Application threads. A common general assumption is
# using 2 per available processor cores - to handle
# incoming requests using one and performing background
# operations using the other.
THREADS_PER_PAGE = 2

# Enable protection agains *Cross-site Request Forgery (CSRF)*
CSRF_ENABLED     = True
CSRF_SESSION_KEY = ""

# Secret key for signing cookies
SECRET_KEY = ""