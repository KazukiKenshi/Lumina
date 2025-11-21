# Lumina Frontend - React Unity WebGL Player

This is a React application that hosts the Lumina Unity WebGL build.

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn

## Installation

1. Install dependencies:
```bash
npm install
```

## Running the Application

Start the development server:
```bash
npm start
```

The application will open in your browser at `http://localhost:3000`.

## Project Structure

```
frontend/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── components/
│   │   ├── UnityPlayer.js  # Unity WebGL player component
│   │   └── UnityPlayer.css # Unity player styles
│   ├── App.js              # Main app component
│   ├── App.css             # App styles
│   ├── index.js            # Entry point
│   └── index.css           # Global styles
├── Build/                  # Unity WebGL build files
│   ├── build.data.br
│   ├── build.framework.js.br
│   ├── build.loader.js
│   └── build.wasm.br
├── TemplateData/           # Unity template assets
│   └── style.css
└── package.json            # Project configuration
```

## Building for Production

Create an optimized production build:
```bash
npm run build
```

The build will be created in the `build/` folder.

## Features

- ✅ React 18 with modern hooks
- ✅ Unity WebGL integration
- ✅ Loading progress indicator
- ✅ Fullscreen support
- ✅ Mobile responsive
- ✅ Error handling
- ✅ Clean, modern UI

## Troubleshooting

If the Unity build doesn't load:
1. Make sure all files in the `Build/` folder are present
2. Check the browser console for errors
3. Ensure the build files are properly served (check MIME types)
4. Try clearing browser cache

## License

This project is part of the Lumina application.
