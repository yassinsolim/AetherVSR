import { defineConfig } from 'vitest/config';

export default defineConfig({
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      // Relative to the Vite root, so no Node path APIs are needed here and
      // the config stays typecheckable without @types/node. Two entry points
      // keep Milestone 2 experiment code out of the baseline harness bundle.
      input: {
        main: 'index.html',
        bench: 'bench.html',
      },
    },
  },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
  },
});
