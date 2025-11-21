const path = require('path');
const fs = require('fs');

module.exports = function(app) {
  // Serve .br files with correct headers
  app.get('*.br', (req, res) => {
    const filePath = path.join(__dirname, '..', 'public', req.path);
    
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      res.status(404).send('File not found');
      return;
    }

    // Set Content-Encoding header for Brotli
    res.setHeader('Content-Encoding', 'br');
    
    // Set appropriate Content-Type based on file extension
    if (req.path.endsWith('.js.br')) {
      res.setHeader('Content-Type', 'application/javascript');
    } else if (req.path.endsWith('.wasm.br')) {
      res.setHeader('Content-Type', 'application/wasm');
    } else if (req.path.endsWith('.data.br')) {
      res.setHeader('Content-Type', 'application/octet-stream');
    }
    
    // Send the file
    res.sendFile(filePath);
  });
};
