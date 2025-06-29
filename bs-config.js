module.exports = {
  server: {
    baseDir: './',
    middleware: {
      1: function noCache(req, res, next) {
        res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
        next();
      }
    }
  },
  files: ['*.html', '*.tsx', '*.wgsl']
};