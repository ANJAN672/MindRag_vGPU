import React from 'react';
import ReactDOM from 'react-dom/client';
import { ChakraProvider, extendTheme, ColorModeScript } from '@chakra-ui/react';
import App from './App';

// Define theme
const theme = extendTheme({
  config: {
    initialColorMode: 'system',
    useSystemColorMode: true,
  },
  fonts: {
    heading: 'Inter, system-ui, sans-serif',
    body: 'Inter, system-ui, sans-serif',
  },
  styles: {
    global: {
      '.markdown-content': {
        h1: {
          fontSize: 'xl',
          fontWeight: 'bold',
          my: 2,
        },
        h2: {
          fontSize: 'lg',
          fontWeight: 'bold',
          my: 2,
        },
        h3: {
          fontSize: 'md',
          fontWeight: 'bold',
          my: 1,
        },
        p: {
          my: 2,
        },
        ul: {
          pl: 5,
          my: 2,
        },
        ol: {
          pl: 5,
          my: 2,
        },
        li: {
          my: 1,
        },
        code: {
          bg: 'gray.100',
          color: 'gray.800',
          p: 0.5,
          borderRadius: 'sm',
          fontFamily: 'monospace',
          _dark: {
            bg: 'gray.700',
            color: 'gray.100',
          },
        },
        pre: {
          bg: 'gray.100',
          color: 'gray.800',
          p: 2,
          borderRadius: 'md',
          overflowX: 'auto',
          fontFamily: 'monospace',
          my: 2,
          _dark: {
            bg: 'gray.700',
            color: 'gray.100',
          },
        },
      },
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ColorModeScript initialColorMode={theme.config.initialColorMode} />
    <ChakraProvider theme={theme}>
      <App />
    </ChakraProvider>
  </React.StrictMode>
);