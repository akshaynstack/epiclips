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
            <body className={`${inter.variable} antialiased`}>
                {children}
            </body>
        </html>
    );
}
