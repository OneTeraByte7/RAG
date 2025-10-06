import { Link } from "react-router-dom"; 

export default function Navbar() {
  return (
    <nav className="px-6 py-2 flex items-center justify-between font-robo bg-black relative z-100">
      {/* Logo / Brand */}
      <div className="text-2xl md:text-3xl font-extrabold tracking-wide  text-yellow-400 cursor-pointer">
        Onmi<span className="text-white">AI</span>
      </div>

      {/* Navigation Links */}
      <div className="hidden md:flex items-center text-white gap-8 text-sm md:text-base font-medium ">
        <a href="#home" className="hover:text-gray-400 transition">Home</a>
        <a href="#features" className="hover:text-gray-400 transition">Features</a>
        <a href="#demo" className="hover:text-gray-400 transition">Demo</a>
        <a href="#about" className="hover:text-gray-400 transition">About</a>
      </div>

      {/* CTA Button */}
      <Link
        to="/chat"
        className="text-sm sm:text-base font-semibold text-yellow-400 hover:text-white transition ease-in-out duration-150"
      >
        Try Chat
      </Link>
    </nav>
  );
}
