/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define THRESH 0.00001


// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    if(is_initialized){
        return;
    }
    
    num_particles = 100;

    // define normal distributions for sensor noise
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // initialize particles
    for (unsigned int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(1.0);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Normal distributions
    normal_distribution<double> n_x(0, std_pos[0]);
    normal_distribution<double> n_y(0, std_pos[1]);
    normal_distribution<double> n_theta(0, std_pos[2]);
    // Add measurements to each particle and random Gaussian noise
    for (int i = 0; i < num_particles; i++) {
        if (fabs(yaw_rate) < THRESH) {  
            particles[i].x += velocity * delta_t * cos(particles[i].theta) + n_x(gen);
            particles[i].y += velocity * delta_t * sin(particles[i].theta) + n_y(gen);
            particles[i].theta += n_theta(gen);
        } 
        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + n_x(gen);
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + n_y(gen);
            particles[i].theta += yaw_rate * delta_t + n_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	 for (unsigned int i = 0; i < observations.size(); i++) {
        // Observation
        LandmarkObs o = observations[i];
    
        // Set the minimum distance to maximum possible.
        double min_dist = numeric_limits<double>::max();
        int map_id = -1;
        
        for (unsigned int j = 0; j < predicted.size(); j++) {
            // Prediction
            LandmarkObs p = predicted[j];
            double dx = o.x - p.x;
            double dy = o.y - p.y;
          
            // Distance between the current observation and prediction.
            double cur_dist = dx * dx + dy * dy;
    
            if (cur_dist < min_dist) {
            min_dist = cur_dist;
            map_id = p.id;
            }
        }
    
        // set the result
        observations[i].id = map_id;
      }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	
    for (int i = 0; i < num_particles; i++) {
        double px = particles[i].x;
        double py = particles[i].y;
        double ptheta = particles[i].theta;
    
        // In range landmark vector
        vector<LandmarkObs> range_landmarks;
        // Map observation
        vector<LandmarkObs> map_observations;
    
        // Find map landmarks within the sensor range
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            // get id and x,y coordinates
            float map_x = map_landmarks.landmark_list[j].x_f;
            float map_y = map_landmarks.landmark_list[j].y_f;
            int map_id = map_landmarks.landmark_list[j].id_i;
          
            if (fabs(map_x - px) <= sensor_range && fabs(map_y - py) <= sensor_range) {
                LandmarkObs range_lanmark;
                range_lanmark.id = map_id;
                range_lanmark.x = map_x;
                range_lanmark.y = map_y;
                range_landmarks.push_back(range_lanmark);
            }
        }
    
        // Transform each observations to map coordinates
        for (unsigned int j = 0; j < observations.size(); j++) {
            double trans_x = cos(ptheta)*observations[j].x - sin(ptheta)*observations[j].y + px;
            double trans_y = sin(ptheta)*observations[j].x + cos(ptheta)*observations[j].y + py;
            LandmarkObs trans_observation;
            trans_observation.id = observations[j].id;
            trans_observation.x = trans_x;
            trans_observation.y = trans_y;
            map_observations.push_back(trans_observation);
        }
    
        // Finds which observations correspond to which landmarks
        dataAssociation(range_landmarks, map_observations);
    
        // reset weight
        particles[i].weight = 1.0;
        // Update the particle weight based on observation between actual device and the particle
        for (unsigned int j = 0; j < map_observations.size(); j++) {
          
            double mo_x, mo_y, in_x, in_y;
            mo_x = map_observations[j].x;
            mo_y = map_observations[j].y;
    
            int mo_i = map_observations[j].id;
    
            for (unsigned int k = 0; k < range_landmarks.size(); k++) {
                if (range_landmarks[k].id == mo_i) {
                    in_x = range_landmarks[k].x;
                    in_y = range_landmarks[k].y;
                }
            }
    
            double s_x = std_landmark[0];
            double s_y = std_landmark[1];
            double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(in_x-mo_x,2)/(2*pow(s_x, 2)) + (pow(in_y-mo_y,2)/(2*pow(s_y, 2))) ) );
    
            // product of this obersvation weight with total observations weight
            particles[i].weight *= obs_w;
        }
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	
	
    uniform_int_distribution<int> uniintdist(0, num_particles-1);
    int index = uniintdist(gen);
    
    // get max weight
    double max_weight = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> unirealdist(0.0, max_weight);
    
    double beta = 0.0;
    
    vector<Particle> resample_particles;
        
    for (unsigned int i = 0; i < num_particles; i++) {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        resample_particles.push_back(particles[index]);
    }
    
    particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
