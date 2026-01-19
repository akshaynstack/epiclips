"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
    ArrowRight,
    Sparkles,
    Video,
    Zap,
    ShieldCheck,
    ArrowUpRight,
    ChevronDown,
    Instagram,
    Twitter,
    Github,
    Linkedin
} from "lucide-react";
import Link from "next/link";

const FeatureCard = ({ icon: Icon, title, description }: any) => (
    <motion.div
        whileHover={{ y: -5 }}
        className="glass-card p-8 group"
    >
        <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500">
            <Icon className="w-6 h-6 text-white" />
        </div>
        <h3 className="text-xl font-bold mb-3">{title}</h3>
        <p className="text-text-secondary leading-relaxed">{description}</p>
    </motion.div>
);

const TrustLogo = ({ name }: { name: string }) => (
    <div className="flex items-center gap-2 text-white/30 grayscale hover:grayscale-0 transition-all cursor-default group">
        <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center group-hover:bg-white/10 transition-colors">
            <Zap className="w-4 h-4" />
        </div>
        <span className="text-lg font-bold tracking-tighter">{name}</span>
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

function PlayIcon({ className }: { className?: string }) {
    return (
        <svg
            viewBox="0 0 24 24"
            fill="currentColor"
            className={className}
            xmlns="http://www.w3.org/2000/svg"
        >
            <path d="M5 3L19 12L5 21V3Z" />
        </svg>
    );
}

function Loader2Icon({ className }: { className?: string }) {
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
            <path d="M12 2v4" />
            <path d="M12 18v4" />
            <path d="M4.93 4.93l2.83 2.83" />
            <path d="M16.24 16.24l2.83 2.83" />
            <path d="M2 12h4" />
            <path d="M18 12h4" />
            <path d="M4.93 19.07l2.83-2.83" />
            <path d="M16.24 7.76l2.83-2.83" />
        </svg>
    );
}

export default function LandingPage() {
    return (
        <div className="min-h-screen bg-black text-white overflow-x-hidden selection:bg-white/30">
            {/* Background Glows */}
            <div className="fixed inset-0 pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/10 blur-[120px] rounded-full animate-pulse-slow" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-500/10 blur-[120px] rounded-full animate-pulse-slow" />
            </div>

            {/* Nav */}
            <nav className="sticky top-0 inset-x-0 h-20 border-b border-white/5 bg-black/50 backdrop-blur-md z-[100]">
                <div className="container-wide h-full flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3 group">
                        <EpiclipsLogo />
                        <span className="font-bold text-xl tracking-tighter">Epiclips</span>
                    </Link>
                    <div className="hidden md:flex items-center gap-8">
                        <Link href="#features" className="text-sm font-medium text-text-secondary hover:text-white transition-colors">Features</Link>
                        <Link href="#how-it-works" className="text-sm font-medium text-text-secondary hover:text-white transition-colors">How it works</Link>
                        <Link href="/docs" className="text-sm font-medium text-text-secondary hover:text-white transition-colors">Docs</Link>
                        <Link href="#faq" className="text-sm font-medium text-text-secondary hover:text-white transition-colors">FAQ</Link>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link href="/editor" className="btn-primary py-2.5 px-6 text-sm">
                            Get Started
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative overflow-hidden">
                <div className="container-wide relative z-10 pt-52 pb-40">
                    <div className="max-w-6xl mx-auto text-center">
                        <motion.div
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, ease: "easeOut" }}
                        >
                            <div className="badge-framer mb-12 mx-auto">
                                <Sparkles className="w-3 h-3 text-white/60" />
                                <span className="text-[10px] uppercase tracking-[0.3em] font-bold">On-Device AI</span>
                            </div>

                            <h1 className="text-7xl md:text-[110px] font-bold tracking-tighter leading-[0.85] mb-16 text-gradient px-4">
                                1 long video.<br className="hidden md:block" /> 10 viral clips.
                            </h1>

                            <p className="text-xl md:text-2xl text-text-secondary max-w-3xl mx-auto mb-20 leading-relaxed font-light px-4">
                                Automatically transform long videos into viral short-form content.
                                <span className="text-white font-medium opacity-100 ml-2">
                                    Privacy First • High Performance • Unlimited Export
                                </span>
                            </p>

                            <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
                                <Link href="/editor" className="btn-primary w-full sm:w-auto px-10 py-5 text-lg">
                                    Start Clipping Now
                                    <ArrowRight className="w-5 h-5" />
                                </Link>
                                <Link href="#how-it-works" className="btn-secondary w-full sm:w-auto px-10 py-5 text-lg font-medium opacity-80 hover:opacity-100">
                                    See How It Works
                                </Link>
                            </div>
                        </motion.div>

                        {/* Hero Image / Mockup */}
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 40 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            transition={{ duration: 1, delay: 0.2 }}
                            className="mt-20 relative"
                        >
                            <div className="absolute inset-0 bg-white/5 blur-[100px] rounded-full animate-pulse-slow" />
                            <div className="relative aspect-video max-w-5xl mx-auto glass-card overflow-hidden shadow-2xl">
                                <div className="absolute inset-0 flex items-center justify-center bg-black/40 group cursor-pointer">
                                    <div className="w-20 h-20 rounded-full bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                                        <PlayIcon className="w-8 h-8 text-white ml-1" />
                                    </div>
                                </div>
                                <img
                                    src="https://images.unsplash.com/photo-1574717024653-61fd2cf4d44d?auto=format&fit=crop&q=80&w=2000"
                                    alt="Epiclips Editor Preview"
                                    className="w-full h-full object-cover"
                                />
                            </div>
                        </motion.div>
                    </div>
                </div>
            </section>

            {/* Trust Section */}
            <section className="py-20 border-y border-white/5 bg-white/[0.01]">
                <div className="container-wide">
                    <p className="text-center text-xs font-bold uppercase tracking-[0.3em] text-white/30 mb-12">Trusted by 10,000+ creators globally</p>
                    <div className="flex flex-wrap justify-center gap-12 md:gap-24 opacity-60">
                        <TrustLogo name="CreatorFlow" />
                        <TrustLogo name="StreamLine" />
                        <TrustLogo name="VibeCast" />
                        <TrustLogo name="Shortsify" />
                        <TrustLogo name="PodPulse" />
                    </div>
                </div>
            </section>

            {/* Features Grid */}
            <section id="features" className="py-32">
                <div className="container-wide">
                    <div className="text-center mb-20">
                        <h2 className="section-title mb-6">AI that understands every<br />pixel of your video</h2>
                        <p className="text-text-secondary text-lg max-w-2xl mx-auto">
                            Built on powerful WebGPU technology, Epiclips processes your videos locally with unprecedented speed and privacy.
                        </p>
                    </div>
                    <div className="grid md:grid-cols-3 gap-8">
                        <FeatureCard
                            icon={Zap}
                            title="Lightning Fast"
                            description="Harness the power of your GPU directly in the browser. No more waiting for slow server uploads."
                        />
                        <FeatureCard
                            icon={ShieldCheck}
                            title="Privacy First"
                            description="Your videos stay on your device. AI transcription and processing happen locally using WebGPU."
                        />
                        <FeatureCard
                            icon={Sparkles}
                            title="Viral Insight"
                            description="Trained on millions of top-performing shorts to identify exactly what hooks viewers."
                        />
                    </div>
                </div>
            </section>

            {/* Workflow Section */}
            <section id="how-it-works" className="py-32 bg-white/[0.01]">
                <div className="container-wide">
                    <div className="grid lg:grid-cols-2 gap-20 items-center">
                        <div>
                            <div className="badge-framer mb-6">Seamless Workflow</div>
                            <h2 className="text-5xl font-bold mb-8 leading-tight">Your video creation process — now on autopilot</h2>
                            <div className="space-y-8">
                                {[
                                    { step: "01", title: "Upload Footage", desc: "Drag and drop any length of video directly into the browser." },
                                    { step: "02", title: "AI Analysis", desc: "Our local AI models analyze speech, sentiment, and visual cues." },
                                    { step: "03", title: "Instant Export", desc: "Get 10+ viral-ready clips in minutes with one click." }
                                ].map((item) => (
                                    <div key={item.step} className="flex gap-6">
                                        <span className="text-white/20 font-bold text-2xl">{item.step}</span>
                                        <div>
                                            <h4 className="font-bold text-xl mb-2">{item.title}</h4>
                                            <p className="text-text-secondary leading-relaxed">{item.desc}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className="relative">
                            <div className="glass-card p-4 aspect-[4/5] relative overflow-hidden group">
                                <div className="absolute inset-0 bg-gradient-to-t from-black to-transparent z-10" />
                                <img
                                    src="https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&q=80&w=800"
                                    alt="Process Mockup"
                                    className="w-full h-full object-cover rounded-2xl group-hover:scale-105 transition-transform duration-1000"
                                />
                                <div className="absolute bottom-8 left-8 right-8 z-20">
                                    <div className="glass-card p-4 flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center">
                                                <Loader2Icon className="w-5 h-5 text-white animate-spin" />
                                            </div>
                                            <span className="font-bold">Extracting viral hooks...</span>
                                        </div>
                                        <span className="text-white/40 text-sm">84%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Solo Engineer Section */}
            <section className="py-32 relative overflow-hidden">
                <div className="container-wide">
                    <div className="glass-card overflow-hidden border-white/[0.05] bg-gradient-to-br from-white/[0.02] to-transparent">
                        <div className="grid md:grid-cols-[400px_1fr] items-center">
                            <div className="relative h-full min-h-[400px] border-r border-white/5">
                                <img
                                    src="/images/akshayn.webp"
                                    alt="Akshay N - Solo Engineer"
                                    className="absolute inset-0 w-full h-full object-cover transition-all duration-700"
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent opacity-60" />
                                <div className="absolute bottom-6 left-6">
                                    <div className="badge-framer bg-white/10 backdrop-blur-md border-white/20 text-white font-bold">
                                        Solo Developer
                                    </div>
                                </div>
                            </div>
                            <div className="p-12 md:p-20 space-y-8">
                                <div>
                                    <h2 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">Engineering behind the <span className="text-gradient">Clipping technology</span></h2>
                                    <p className="text-xl text-text-secondary leading-relaxed">
                                        Epiclips isn&apos;t built by a corporate army. It&apos;s the result of a singular obsession with pushing the boundaries of what&apos;s possible with WebGPU and local AI.
                                    </p>
                                </div>

                                <div className="space-y-6">
                                    <div className="flex items-center gap-4">
                                        <div className="w-12 h-12 rounded-full border border-white/10 flex items-center justify-center bg-white/5">
                                            <Sparkles className="w-5 h-5 text-white/60" />
                                        </div>
                                        <div>
                                            <h4 className="font-bold">AKSHAY N</h4>
                                            <p className="text-sm text-text-muted italic">Founder & Solo Engineer</p>
                                        </div>
                                    </div>

                                    <blockquote className="border-l-2 border-white/10 pl-6 text-text-secondary italic">
                                        &quot;My goal was to democratize viral content creation by bringing high-end AI processing directly to your browser. No servers, no latency, just pure engineering.&quot;
                                    </blockquote>
                                    <div className="flex gap-4">
                                        <Link
                                            href="https://x.com/akshaynceo"
                                            target="_blank"
                                            className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 hover:border-white/20 transition-all group/icon"
                                        >
                                            <Twitter className="w-5 h-5 text-text-secondary group-hover/icon:text-white transition-colors" />
                                        </Link>
                                        <Link
                                            href="https://github.com/akshaynstack"
                                            target="_blank"
                                            className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 hover:border-white/20 transition-all group/icon"
                                        >
                                            <Github className="w-5 h-5 text-text-secondary group-hover/icon:text-white transition-colors" />
                                        </Link>
                                        <Link
                                            href="https://www.linkedin.com/in/akshaynstack/"
                                            target="_blank"
                                            className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 hover:border-white/20 transition-all group/icon"
                                        >
                                            <Linkedin className="w-5 h-5 text-text-secondary group-hover/icon:text-white transition-colors" />
                                        </Link>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Footer */}
            <section className="py-32">
                <div className="container-wide text-center">
                    <div className="glass-card py-20 px-8 relative overflow-hidden overflow-hidden">
                        <div className="absolute inset-0 bg-radial-glow opacity-50" />
                        <div className="relative z-10">
                            <h2 className="text-5xl md:text-7xl font-bold mb-8 tracking-tighter">Ready to go viral?</h2>
                            <p className="text-xl text-text-secondary max-w-xl mx-auto mb-12">
                                Join thousands of creators using Epiclips to scale their short-form presence.
                            </p>
                            <Link href="/editor" className="btn-primary">
                                Get Started Now
                                <ArrowRight className="w-5 h-5" />
                            </Link>
                        </div>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-20 border-t border-white/5">
                <div className="container-wide">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-12 mb-20">
                        <div className="col-span-2 md:col-span-1">
                            <Link href="/" className="flex items-center gap-3 mb-6 group">
                                <EpiclipsLogo size="w-6 h-6" />
                                <span className="font-bold text-lg tracking-tighter">Epiclips</span>
                            </Link>
                            <p className="text-sm text-text-secondary leading-relaxed">
                                Empowering creators with on-device AI for viral short-form content.
                            </p>
                        </div>
                        <div>
                            <h5 className="font-bold text-sm uppercase tracking-widest mb-6">Product</h5>
                            <ul className="space-y-4 text-sm text-text-secondary">
                                <li><Link href="#" className="hover:text-white transition-colors">Features</Link></li>
                                <li><Link href="#" className="hover:text-white transition-colors">Enterprise</Link></li>
                                <li><Link href="#" className="hover:text-white transition-colors">Showcase</Link></li>
                            </ul>
                        </div>
                        <div>
                            <h5 className="font-bold text-sm uppercase tracking-widest mb-6">Company</h5>
                            <ul className="space-y-4 text-sm text-text-secondary">
                                <li><Link href="#" className="hover:text-white transition-colors">About</Link></li>
                                <li><Link href="#" className="hover:text-white transition-colors">Blog</Link></li>
                                <li><Link href="#" className="hover:text-white transition-colors">Careers</Link></li>
                            </ul>
                        </div>
                        <div>
                            <h5 className="font-bold text-sm uppercase tracking-widest mb-6">Social</h5>
                            <div className="flex gap-4">
                                <Link
                                    href="https://x.com/akshaynceo"
                                    target="_blank"
                                    className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center hover:bg-white/10 transition-colors"
                                >
                                    <Twitter className="w-4 h-4" />
                                </Link>
                                <Link
                                    href="https://github.com/akshaynstack"
                                    target="_blank"
                                    className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center hover:bg-white/10 transition-colors"
                                >
                                    <Github className="w-4 h-4" />
                                </Link>
                                <Link
                                    href="https://www.linkedin.com/in/akshaynstack/"
                                    target="_blank"
                                    className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center hover:bg-white/10 transition-colors"
                                >
                                    <Linkedin className="w-4 h-4" />
                                </Link>
                            </div>
                        </div>
                    </div>
                    <div className="flex flex-col md:flex-row items-center justify-between pt-12 border-t border-white/5 text-xs text-text-muted gap-6">
                        <p>© 2024 Epiclips Labs. Built with WebGPU.</p>
                        <div className="flex gap-8">
                            <Link href="#" className="hover:text-white">Privacy Policy</Link>
                            <Link href="#" className="hover:text-white">Terms of Service</Link>
                            <Link href="#" className="hover:text-white">Cookie Policy</Link>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
}
