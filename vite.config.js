/// <reference types="vitest" />
import * as path from 'path'
import {resolve} from 'path'
import {defineConfig} from 'vite';
// For wasm inclusion
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
// For wasm generated by wasm-pack
import wasmPack from 'vite-plugin-wasm-pack';

export default defineConfig({
    build: {
        minify: 'esbuild',
        lib: {
            // Could also be a dictionary or array of multiple entry points
            entry: resolve(__dirname, 'pkg/fits3.js'),
            name: 'fits3',
            formats: ["umd", "es"],
            // the proper extensions will be added
            fileName: 'fits3',
        },
        //rollupOptions: {},
        format: "es",
        //target: ["es2015", "chrome58", "edge18", "firefox57", "node12", "safari11"],
        //target: ["es2015"],
        // Relative to the root
        outDir: resolve(__dirname, 'dist'),
    },
    //publicDir: resolve(__dirname, 'src/img'),
    plugins: [
        wasm(),
        wasmPack(__dirname),
        topLevelAwait(),
    ],
    resolve: {
        alias: [
            {find: '@', replacement: path.resolve(__dirname, '/src')},
            {find: '#', replacement: path.resolve(__dirname, '/tests/unit')},
            {find: '$', replacement: path.resolve(__dirname, '/tests/e2e')}
        ],
    },
    test: {
        globals: true,
        environment: 'happy-dom',
        include: [
            'tests/unit/**/*.{test,spec}.{js,ts}'
        ],
        deps: {
            inline: ['pkg'],
        },
    },
    server: {
        open: 'index.html',
    },
});