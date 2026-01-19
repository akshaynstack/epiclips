"use client";

import { motion } from "framer-motion";
import {
    ArrowRight,
    Terminal,
    Copy,
    Check,
    ChevronRight,
    Book,
    Code,
    Cpu,
    Server
} from "lucide-react";
import Link from "next/link";
import { useState } from "react";

const CopyBlock = ({ text, label }: { text: string; label?: string }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="my-4">
            {label && <div className="text-xs font-bold text-text-secondary uppercase mb-2 tracking-wider">{label}</div>}
            <div className="glass-card bg-black/50 border border-white/10 p-4 rounded-xl relative group">
                <div className="font-mono text-sm text-gray-300 overflow-x-auto pr-10">
                    <span className="select-none text-gray-500 mr-2">$</span>
                    {text}
                </div>
                <button
                    onClick={handleCopy}
                    className="absolute right-3 top-3 p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors border border-white/5 text-gray-400 hover:text-white"
                >
                    {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                </button>
            </div>
        </div>
    );
};

const StepCard = ({ number, title, children }: { number: string; title: string; children: React.ReactNode }) => (
    <div className="relative pl-8 md:pl-0">
        <div className="glass-card p-6 md:p-8 border border-white/5 bg-white/[0.02]">
            <div className="flex items-start gap-4">
                <div className="hidden md:flex w-8 h-8 rounded-full bg-white/5 border border-white/10 items-center justify-center shrink-0 text-sm font-bold text-white/50">
                    {number}
                </div>
                <div className="w-full">
                    <h3 className="text-xl font-bold mb-4 flex items-center gap-3">
                        <span className="md:hidden text-white/50">{number}.</span>
                        {title}
                    </h3>
                    <div className="text-text-secondary space-y-4">
                        {children}
                    </div>
                </div>
            </div>
        </div>
    </div>
);

const EpiclipsLogo = ({ size = "w-8 h-8" }: { size?: string }) => (
    <div className={`${size} relative group/logo flex items-center justify-center`}>
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-full h-full text-white transition-transform duration-500 group-hover/logo:scale-110"
        >
            <path d="M6 5h12" />
            <path d="M4 12h10" />
            <path d="M12 19h8" />
        </svg>
    </div>
);

export default function DocsPage() {
    return (
        <div className="min-h-screen bg-black text-white selection:bg-white/30">
            {/* Background Glows */}
            <div className="fixed inset-0 pointer-events-none">
                <div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/5 blur-[120px] rounded-full animate-pulse-slow" />
                <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-500/5 blur-[120px] rounded-full animate-pulse-slow" />
            </div>

            {/* Nav */}
            <nav className="sticky top-0 inset-x-0 h-20 border-b border-white/5 bg-black/50 backdrop-blur-md z-[100]">
                <div className="container-wide h-full flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3 group">
                        <EpiclipsLogo />
                        <span className="font-bold text-xl tracking-tighter">Epiclips</span>
                        <span className="px-2 py-0.5 rounded-full bg-white/10 text-[10px] font-bold uppercase tracking-widest text-white/60">Docs</span>
                    </Link>
                    <div className="hidden md:flex items-center gap-8">
                        <Link href="/#features" className="text-sm font-medium text-text-secondary hover:text-white transition-colors">Features</Link>
                        <Link href="/#how-it-works" className="text-sm font-medium text-text-secondary hover:text-white transition-colors">How it works</Link>
                        <Link href="https://github.com/akshaynstack/epiclips" target="_blank" className="text-sm font-medium text-text-secondary hover:text-white transition-colors">GitHub</Link>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link href="/" className="btn-secondary py-2 px-4 text-sm">
                            Back to Home
                        </Link>
                        <Link href="/editor" className="btn-primary py-2 px-4 text-sm">
                            Open App
                        </Link>
                    </div>
                </div>
            </nav>

            <div className="container-wide max-w-5xl mx-auto py-20">
                {/* Hero Section */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center mb-20"
                >
                    <div className="badge-framer mb-8 mx-auto">
                        <Book className="w-3 h-3 text-white/60" />
                        <span className="text-[10px] uppercase tracking-[0.3em] font-bold">Documentation</span>
                    </div>

                    <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-8 text-gradient">
                        Setup & Installation
                    </h1>

                    <p className="text-xl text-text-secondary max-w-2xl mx-auto mb-12 leading-relaxed">
                        Get Epiclips running locally on your machine in minutes. Use our recommended Docker setup or configure manually with uv.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <Link
                            href="https://github.com/akshaynstack/epiclips"
                            target="_blank"
                            className="btn-primary py-4 px-8 text-lg min-w-[200px]"
                        >
                            <Terminal className="w-5 h-5" />
                            Clone Repository
                        </Link>
                        <Link
                            href="#quick-start"
                            className="btn-secondary py-4 px-8 text-lg min-w-[200px]"
                        >
                            Read Guide
                            <ChevronDownIcon className="w-5 h-5" />
                        </Link>
                    </div>
                </motion.div>

                <div className="grid lg:grid-cols-[250px_1fr] gap-12">
                    {/* Sidebar */}
                    <aside className="hidden lg:block sticky top-24 h-fit space-y-8">
                        <div>
                            <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                                <Terminal className="w-4 h-4 text-blue-400" />
                                Quick Start
                            </h4>
                            <ul className="space-y-2 border-l border-white/10 pl-4 text-sm">
                                <li><a href="#prerequisites" className="text-text-secondary hover:text-white block py-1 transition-colors">Prerequisites</a></li>
                                <li><a href="#docker-setup" className="text-text-secondary hover:text-white block py-1 transition-colors">Docker Setup</a></li>
                                <li><a href="#manual-setup" className="text-text-secondary hover:text-white block py-1 transition-colors">Manual Setup</a></li>
                                <li><a href="#env-vars" className="text-text-secondary hover:text-white block py-1 transition-colors">Environment Variables</a></li>
                            </ul>
                        </div>
                    </aside>

                    {/* Content */}
                    <div className="space-y-12" id="quick-start">
                        {/* Prerequisites */}
                        <section id="prerequisites">
                            <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
                                <Cpu className="w-6 h-6 text-purple-400" />
                                Prerequisites
                            </h2>
                            <div className="glass-card p-6 border border-white/5">
                                <ul className="space-y-3">
                                    <li className="flex items-center gap-3 text-text-secondary">
                                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                        Python 3.11+
                                    </li>
                                    <li className="flex items-center gap-3 text-text-secondary">
                                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                        Node.js 18+
                                    </li>
                                    <li className="flex items-center gap-3 text-text-secondary">
                                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                        FFmpeg
                                    </li>
                                    <li className="flex items-center gap-3 text-text-secondary">
                                        <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
                                        Docker (Recommended)
                                    </li>
                                </ul>
                            </div>
                        </section>

                        {/* Docker Setup */}
                        <section id="docker-setup">
                            <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
                                <Server className="w-6 h-6 text-blue-400" />
                                Option 1: Docker (Recommended)
                            </h2>
                            <div className="space-y-6">
                                <StepCard number="01" title="Clone the repository">
                                    <p>Start by cloning only the necessary files to your local machine.</p>
                                    <CopyBlock text="git clone https://github.com/akshaynstack/epiclips.git" />
                                    <CopyBlock text="cd epiclips" />
                                </StepCard>

                                <StepCard number="02" title="Configure Environment">
                                    <p>Copy the example environment file and update it with your API keys.</p>
                                    <CopyBlock text="cp .env.example .env" />
                                </StepCard>

                                <StepCard number="03" title="Run with Docker Compose">
                                    <p>Start the entire stack (backend + frontend) with a single command.</p>
                                    <CopyBlock text="docker-compose up -d" />
                                    <p className="mt-4 text-sm text-text-secondary border-t border-white/5 pt-4">
                                        The API will be available at <span className="text-white font-mono">http://localhost:8000</span> and the app at <span className="text-white font-mono">http://localhost:3000</span>
                                    </p>
                                </StepCard>
                            </div>
                        </section>

                        {/* Manual Setup */}
                        <section id="manual-setup">
                            <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
                                <Code className="w-6 h-6 text-orange-400" />
                                Option 2: Manual Setup
                            </h2>
                            <div className="space-y-8">
                                <div className="border border-white/5 bg-white/[0.02] rounded-2xl overflow-hidden">
                                    <div className="bg-white/5 px-6 py-4 border-b border-white/5 font-bold">Backend Setup</div>
                                    <div className="p-6 md:p-8 space-y-6">
                                        <div>
                                            <p className="mb-4 text-text-secondary">1. Initialize and install dependencies using <code>uv</code>:</p>
                                            <CopyBlock text="uv sync --python 3.11" />
                                        </div>
                                        <div>
                                            <p className="mb-4 text-text-secondary">2. Run the development server:</p>
                                            <CopyBlock text="uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" />
                                        </div>
                                    </div>
                                </div>

                                <div className="border border-white/5 bg-white/[0.02] rounded-2xl overflow-hidden">
                                    <div className="bg-white/5 px-6 py-4 border-b border-white/5 font-bold">Frontend Setup</div>
                                    <div className="p-6 md:p-8 space-y-6">
                                        <div>
                                            <p className="mb-4 text-text-secondary">1. Navigate to frontend directory and install dependencies:</p>
                                            <CopyBlock text="cd frontend && npm install" />
                                        </div>
                                        <div>
                                            <p className="mb-4 text-text-secondary">2. Start Next.js development server:</p>
                                            <CopyBlock text="npm run dev" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </section>

                        {/* Env Vars */}
                        <section id="env-vars">
                            <h2 className="text-3xl font-bold mb-6">Environment Variables</h2>
                            <div className="glass-card p-6 md:p-8 overflow-hidden">
                                <p className="mb-6 text-text-secondary">Create a <code className="text-white bg-white/10 px-1.5 py-0.5 rounded">.env</code> file in the root directory:</p>
                                <div className="bg-black/50 border border-white/10 rounded-lg p-4 font-mono text-sm text-gray-400 overflow-x-auto">
                                    <pre>{`# AI Services
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=google/gemini-flash-1.5-8b
GROQ_API_KEY=your_key

# Optional
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
WEBHOOK_URL=...`}</pre>
                                </div>
                            </div>
                        </section>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <footer className="py-12 border-t border-white/5 mt-20">
                <div className="container-wide text-center text-text-secondary text-sm">
                    <p>Need help? Check out the <a href="https://github.com/akshaynstack/epiclips/issues" className="text-white hover:underline">GitHub Issues</a>.</p>
                </div>
            </footer>
        </div>
    );
}

function ChevronDownIcon({ className }: { className?: string }) {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={className}
        >
            <path d="M6 9l6 6 6-6" />
        </svg>
    );
}
