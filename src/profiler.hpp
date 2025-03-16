//
// Created for profiling performance bottlenecks
//

#ifndef KAIZEN_PROFILER_HPP
#define KAIZEN_PROFILER_HPP

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include "timer.hpp"

// Profiler class to measure execution time of different code segments
class Profiler {
public:
    Profiler() {}
    
    // Start timing a specific segment
    void start(const std::string& segment_name) {
        if (timers.find(segment_name) == timers.end()) {
            timers[segment_name] = timer();
            call_counts[segment_name] = 0;
        }
        timers[segment_name].start();
        call_counts[segment_name]++;
    }
    
    // Stop timing a specific segment
    void stop(const std::string& segment_name) {
        if (timers.find(segment_name) != timers.end()) {
            timers[segment_name].stop();
        }
    }
    
    // Print report of all timed segments
    void print_report() const {
        std::cout << "\n===== PERFORMANCE PROFILING REPORT =====\n";
        std::cout << std::setw(30) << std::left << "Segment" 
                  << std::setw(15) << std::right << "Total Time (s)" 
                  << std::setw(15) << "Calls" 
                  << std::setw(15) << "Avg Time (ms)" << std::endl;
        std::cout << std::string(75, '-') << std::endl;
        
        // Collect results for sorting
        std::vector<std::pair<std::string, double>> sorted_segments;
        for (const auto& entry : timers) {
            sorted_segments.push_back({entry.first, entry.second.elapse_sec()});
        }
        
        // Sort by total time (descending)
        std::sort(sorted_segments.begin(), sorted_segments.end(), 
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        double total_time = 0.0;
        for (const auto& segment : sorted_segments) {
            const auto& name = segment.first;
            const double time = segment.second;
            const int count = call_counts.at(name);
            const double avg_ms = (count > 0) ? (time * 1000.0 / count) : 0.0;
            
            std::cout << std::setw(30) << std::left << name 
                      << std::setw(15) << std::fixed << std::setprecision(6) << std::right << time
                      << std::setw(15) << count 
                      << std::setw(15) << std::fixed << std::setprecision(3) << avg_ms << std::endl;
            
            total_time += time;
        }
        
        std::cout << std::string(75, '-') << std::endl;
        std::cout << std::setw(30) << std::left << "TOTAL" 
                  << std::setw(15) << std::fixed << std::setprecision(6) << std::right << total_time << std::endl;
        std::cout << "\nBottleneck Analysis:\n";
        
        if (!sorted_segments.empty()) {
            const auto& bottleneck = sorted_segments.front();
            std::cout << "Primary bottleneck: " << bottleneck.first << " (" 
                      << std::fixed << std::setprecision(2) 
                      << (bottleneck.second / total_time * 100.0) << "% of total time)" << std::endl;
        }
    }
    
    // Reset all timers
    void reset() {
        for (auto& timer_pair : timers) {
            timer_pair.second.clear();
        }
        call_counts.clear();
    }
    
private:
    std::map<std::string, timer> timers;
    std::map<std::string, int> call_counts;
};

// Global profiler instance
extern Profiler g_profiler;

// Macro for easy profiling of code blocks
#define PROFILE_SCOPE(name) ProfilerScope profiler_scope_##__LINE__(name)

// Helper class for automatic profiling using RAII
class ProfilerScope {
public:
    ProfilerScope(const std::string& name) : name_(name) {
        g_profiler.start(name_);
    }
    
    ~ProfilerScope() {
        g_profiler.stop(name_);
    }
    
private:
    std::string name_;
};

#endif // KAIZEN_PROFILER_HPP 