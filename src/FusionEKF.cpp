#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
      0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
      0, 1, 0, 0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
  /**
   * Initialization
   */
  if (!is_initialized_)
  {

    // Initialize the state ekf_.x_ with the first measurement.

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    //Create state covariance matrix
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

    //initial transition matrix
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
      // Convert radar from polar to cartesian coordinates
      // and initialize state.

      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);

      float p_x = rho * cos(phi);
      float p_y = rho * sin(phi);
      float v_x = rho_dot * cos(phi);
      float v_y = rho_dot * sin(phi);

      ekf_.x_ << p_x, p_y, v_x, v_y;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
      // Initialize state.

      float p_x = measurement_pack.raw_measurements_(0);
      float p_y = measurement_pack.raw_measurements_(1);
      float v_x = 0.0;
      float v_y = 0.0;

      ekf_.x_ << p_x, p_y, v_x, v_y;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  // Update the state transition matrix F according to the new elapsed time.
  // Time is measured in seconds.

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

  // Update timestamp
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Update the process noise covariance matrix.
  // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
  float noise_ax = 9.0;
  float noise_ay = 9.0;

  // set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
      0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
      dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    // Radar updates
    // y=z−h(x')
    Tools calculate;

    // Update h()
    ekf_.H_ = calculate.CalculateJacobian(ekf_.x_);
    // Update R
    ekf_.R_ = R_radar_;
    // Update using EKF equation
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else
  {
    // Laser updates
    // Update H
    ekf_.H_ = H_laser_;
    // Update R
    ekf_.R_ = R_laser_;
    // Update using KF equation
    ekf_.Update(measurement_pack.raw_measurements_);
  }
}