The 'bun.exe' file is a placeholder for the binary file, which
is replaced by Bun's 'postinstall' script. For this to work, make
sure that you do not use --ignore-scripts while installing.

The postinstall script is responsible for linking the binary file
directly into 'node_modules/.bin' and avoiding a Node.js wrapper
script being called on every invocation of 'bun'. If this wasn't
done, Bun would seem to be slower than Node.js, because it would
be executing a copy of Node.js every time!

Unfortunately, it is not possible to fix all cases on all platforms
without *requiring* a postinstall script.
