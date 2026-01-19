import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
    subsets: ["latin"],
    variable: "--font-inter",
});

export const metadata: Metadata = {
    title: "Epiclips | AI-Powered Video Clipping",
    description: "Transform long videos into viral short-form content with AI. Browser-based, privacy-first, powered by WebGPU.",
    keywords: ["AI video editor", "clip generator", "short form content", "viral clips", "WebGPU"],
    openGraph: {
        title: "Epiclips | AI-Powered Video Clipping",
        description: "Transform long videos into viral short-form content with AI",
        type: "website",
    },
    icons: {
        icon: "/favicon.png",
    },
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className="dark">
            <body className={`${inter.variable} antialiased`} suppressHydrationWarning>
                {/* Development Notice Banner */}
                <div className="relative z-[110] bg-gradient-to-r from-amber-500 via-orange-600 to-amber-500 text-white text-center py-2.5 px-4 text-xs font-bold uppercase tracking-widest shadow-2xl border-b border-white/10">
                    <span className="inline-flex items-center gap-2">
                        <svg className="w-3 h-3 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                        Development Preview: Some features may be unstable
                    </span>
                </div>
                {children}
            </body>
        </html>
    );
}
