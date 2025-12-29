/** @type {import('next').NextConfig} */
const nextConfig = {
    // Enable standalone output for Docker deployment
    output: 'standalone',

    // Disable Turbopack as FFmpeg.wasm needs custom Webpack config
    // Note: Turbopack is often default in Next 15 dev, but custom webpack disables it automatically
    webpack: (config, { isServer, webpack }) => {
        // Globally ignore Node.js specific modules used by Transformers.js
        // This prevents SSR errors like "sharp.format is not a function"
        config.resolve.alias = {
            ...config.resolve.alias,
            'sharp': false,
            'onnxruntime-node': false,
        };

        if (!isServer) {
            config.resolve.fallback = {
                ...config.resolve.fallback,
                fs: false,
                path: false,
                crypto: false,
            };
        }

        // Fix for "Unexpected character ''" error:
        // Treat binary .node files as assets instead of trying to parse them as JS
        config.module.rules.push({
            test: /\.node$/,
            type: 'asset/resource',
        });

        // Fixed: Module not found: Can't resolve <dynamic>
        config.experiments = {
            ...config.experiments,
            topLevelAwait: true,
            asyncWebAssembly: true,
        };
        return config;
    },
    // Headers for SharedArrayBuffer (required for FFmpeg multithreading)
    async headers() {
        return [
            {
                source: '/:path*',
                headers: [
                    {
                        key: 'Cross-Origin-Opener-Policy',
                        value: 'same-origin',
                    },
                    {
                        key: 'Cross-Origin-Embedder-Policy',
                        value: 'require-corp',
                    },
                ],
            },
        ];
    },
};

export default nextConfig;
