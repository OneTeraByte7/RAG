import React, { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import bgImg1 from "../assets/bgImg01.png";
import bgImg2 from "../assets/bgImg02.png";
import grapImg1 from "../assets/graphic01.png";
import { Cpu, Zap, Globe, Shield } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const LandingPage = () => {
  // ✅ Refs
  const heroTitleRef = useRef(null);
  const heroSubRef = useRef(null);
  const heroBtnRef = useRef(null);
  const featuresRef = useRef([]);
  const stepsRef = useRef([]);

  useEffect(() => {
    // ✅ Hero animations
    gsap.fromTo(
      heroTitleRef.current,
      { xPercent: 80, opacity: 0 },
      { xPercent: 0, opacity: 1, duration: 1, ease: "power3.out" }
    );

    gsap.fromTo(
      heroSubRef.current,
      { xPercent: 50, opacity: 0 },
      { xPercent: 0, opacity: 1, duration: 1, delay: 1, ease: "power3.out" }
    );



    // ✅ Features Section
    gsap.fromTo(
      featuresRef.current,
      { yPercent: 80, opacity: 0 },
      {
        yPercent: 0,
        opacity: 1,
        duration: 1,
        stagger: 0.2,
        ease: "power3.out",
        scrollTrigger: {
          trigger: featuresRef.current[0],
          start: "top 85%",
        },
      }
    );

    // ✅ How It Works Section
    gsap.fromTo(
      stepsRef.current,
      { opacity: 0 },
      {
        opacity: 1,
        duration: 1,
        stagger: 0.3,
        ease: "power3.out",
        scrollTrigger: {
          trigger: stepsRef.current[0],
          start: "top 85%",
        },
      }
    );
  }, []);

  return (
    <main className="font-robo">
      <Navbar />
      {/* ✅ Hero Section */}
      <section
        className="w-full min-h-screen flex flex-col items-start justify-center text-left px-6 relative"
        style={{
          backgroundImage: `url(${bgImg1})`,
          backgroundSize: "cover",
        }}
      >
        {/* Graphic overlay */}
        <div
          className="absolute bg-amber-500 md:none md:w-[50%] h-full top-0 right-0 z-1 border-r-4"
          style={{
            backgroundImage: `url(${grapImg1})`,
            backgroundSize: "cover",
            clipPath:
              "polygon(0% 75%, 20% 75%, 20% 35%, 40% 35%, 40% 15%, 60% 15%, 60% 0%, 100% 0%, 100% 100%, 0% 100%)",
          }}
        ></div>

        {/* Headline */}
        <h1
          ref={heroTitleRef}
          className="text-4xl md:text-5xl lg:text-6xl font-bold text-white font-mont z-3"
        >
          <span className="text-yellow-400">AI</span> Assistant That <br />
          <span className="text-yellow-400">Connects</span> All <br /> Your
          <span className="text-yellow-400"> Digital Worlds</span>
        </h1>

        {/* Subheadline */}
        <p
          ref={heroSubRef}
          className="mt-4 text-base md:text-xl text-white max-w-4xl z-3"
        >
          Seamlessly search and interact with your documents, images, audio, and
          more — all in one place. Boost productivity, save time, and unlock
          insights instantly.
        </p>

        {/* Buttons */}
        <div className="mt-6 flex justify-center z-3">
          <Link to="/chat">
            <button
              className="bg-yellow-400 text-black px-10 py-2 rounded-tr-2xl border-2 border-yellow-400 text-sm sm:text-base font-semibold hover:bg-black hover:text-yellow-400 transition hover:border-2 ease-in-out duration-150"
            >
              Get Started
            </button>
          </Link>
        </div>
      </section>

      {/* ✅ Features Section */}
      <section
        className="w-full py-16 bg-black text-white text-center px-6"
        style={{
          backgroundImage: `url(${bgImg2})`,
          backgroundSize: "cover",
        }}
      >
        <h2 className="text-3xl md:text-4xl font-bold mb-4">
          Why Choose <span className="text-yellow-400">Our AI?</span>
        </h2>
        <p className="text-gray-300 max-w-3xl mx-auto mb-12">
          Designed to power your digital life — from boosting productivity to
          securing your data, our AI assistant does it all.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mx-auto">
          {[
            {
              icon: <Cpu className="mx-auto text-yellow-400 w-10 h-10 mb-4" />,
              title: "Smart Processing",
              desc: "Harness the power of AI to organize and analyze your data in seconds.",
            },
            {
              icon: <Zap className="mx-auto text-yellow-400 w-10 h-10 mb-4" />,
              title: "Lightning Fast",
              desc: "Get results instantly with optimized AI performance and speed.",
            },
            {
              icon: <Globe className="mx-auto text-yellow-400 w-10 h-10 mb-4" />,
              title: "Global Access",
              desc: "Connect with your files and tools from anywhere in the world.",
            },
            {
              icon: <Shield className="mx-auto text-yellow-400 w-10 h-10 mb-4" />,
              title: "Secure & Reliable",
              desc: "Your data stays safe with enterprise-grade encryption & privacy.",
            },
          ].map((feature, i) => (
            <div
              key={i}
              ref={(el) => (featuresRef.current[i] = el)}
              className="border-2 border-yellow-400 rounded-2xl p-6 shadow-lg hover:shadow-yellow-400/40 transition"
            >
              {feature.icon}
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ✅ How It Works Section */}
      <section className="w-full py-20 bg-yellow-400 px-6 text-black">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            How <span className="text-white">It Works</span>
          </h2>
          <p className="max-w-3xl mx-auto mb-12">
            Our AI assistant is simple, fast, and powerful. Here’s how you can
            start transforming your workflow in just 3 steps.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            {[
              {
                step: "1",
                title: "Upload",
                desc: "Add your documents, images, or files securely into the platform.",
              },
              {
                step: "2",
                title: "Ask",
                desc: "Simply type or speak your question, and our AI assistant gets to work.",
              },
              {
                step: "3",
                title: "Discover",
                desc: "Instantly see results, insights, and actions you can take right away.",
              },
            ].map((item, i) => (
              <div
                key={i}
                ref={(el) => (stepsRef.current[i] = el)}
                className="p-6 bg-black rounded-2xl border border-yellow-400 shadow-lg hover:shadow-yellow-400/40 transition"
              >
                <div className="w-12 h-12 flex items-center justify-center mx-auto mb-4 rounded-full bg-yellow-400 text-black font-bold">
                  {item.step}
                </div>
                <h3 className="text-xl font-semibold mb-2 text-white">
                  {item.title}
                </h3>
                <p className="text-gray-400">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
};

export default LandingPage;
