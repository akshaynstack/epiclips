import { Film } from 'lucide-react'

export default {
    logo: (
        <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700 }}>
            <Film size={24} />
            Epiclips
        </span>
    ),
    project: {
        link: 'https://github.com/akshaynstack/epiclips',
    },
    docsRepositoryBase: 'https://github.com/akshaynstack/epiclips/tree/main/docs',
    useNextSeoProps() {
        return {
            titleTemplate: '%s – Epiclips'
        }
    },
    head: (
        <>
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <meta property="og:title" content="Epiclips - AI Video Clipping" />
            <meta property="og:description" content="Open-source AI-powered video clipping tool" />
        </>
    ),
    banner: {
        key: 'beta-release',
        text: (
            <a href="https://github.com/akshaynstack/epiclips" target="_blank">
                Epiclips is now open source! Star us on GitHub →
            </a>
        ),
    },
    sidebar: {
        titleComponent({ title, type }) {
            if (type === 'separator') {
                return <span className="cursor-default">{title}</span>
            }
            return <>{title}</>
        },
        defaultMenuCollapseLevel: 1,
        toggleButton: true,
    },
    footer: {
        text: (
            <span>
                MIT {new Date().getFullYear()} ©{' '}
                <a href="https://github.com/akshaynstack" target="_blank">
                    Akshay
                </a>
            </span>
        ),
    },
    primaryHue: 270,
    primarySaturation: 100,
}
