#ifndef LIDAROBJECTFACTOR_H
#define LIDAROBJECTFACTOR_H

#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <assert.h>
#include <cmath>
#include <fstream>
#include "utils/math_tools.h"
#include "utils/utility.h"

struct ConstAccFactorAuto
{
    ConstAccFactorAuto(double _dt, Eigen::MatrixXd _Cov_proir, double _motion_weight) : dt(_dt), Cov_proir(_Cov_proir), motion_weight(_motion_weight) {}

    template <typename T>
    bool operator()(const T *const pose_pre, const T *const linearVel_pre, const T *const angularVel_pre, const T *acc_pre,
                    const T *const pose_now, const T *const linearVel_now, const T *const angularVel_now, const T *const acc_now, T *residuals) const
    {
        // Note: P:W系   V:W系   W:b系   A:b系
        Eigen::Matrix<T, 3, 1> P_pre(pose_pre[0], pose_pre[1], pose_pre[2]);
        Eigen::Matrix<T, 3, 1> V_pre(linearVel_pre[0], linearVel_pre[1], T(0));
        Eigen::Matrix<T, 3, 1> W_pre(angularVel_pre[0], angularVel_pre[1], angularVel_pre[2]);
        Eigen::Matrix<T, 3, 1> A_pre(acc_pre[0], acc_pre[1], T(0));
        Eigen::Quaternion<T> Q_pre{pose_pre[6], pose_pre[3], pose_pre[4], pose_pre[5]};
        Eigen::Matrix<T, 3, 3> R_pre = Q_pre.toRotationMatrix();

        Eigen::Matrix<T, 3, 1> P_now(pose_now[0], pose_now[1], pose_now[2]);
        Eigen::Matrix<T, 3, 1> V_now(linearVel_now[0], linearVel_now[1], T(0));
        Eigen::Matrix<T, 3, 1> W_now(angularVel_now[0], angularVel_now[1], angularVel_now[2]);
        Eigen::Matrix<T, 3, 1> A_now(acc_now[0], acc_now[1], T(0));
        Eigen::Quaternion<T> Q_now{pose_now[6], pose_now[3], pose_now[4], pose_now[5]};

        Eigen::Matrix<T, 3, 1> p_error = P_now - (P_pre + V_pre * dt + Q_pre * (A_pre * (0.5 * dt * dt)));
        Eigen::Quaternion<T> delta_q = Q_now.conjugate() * (Q_pre * Utility::deltaQ(W_pre * dt));
        Eigen::Matrix<T, 3, 1> v_error = V_now - (V_pre + Q_pre * (A_pre * dt));
        Eigen::Matrix<T, 3, 1> w_error = W_now - W_pre;
        Eigen::Matrix<T, 3, 1> a_error = A_now - A_pre;

        Eigen::Matrix<T, 15, 1> Residuals;

        Residuals(0, 0) = T(motion_weight)*p_error[0];
        Residuals(1, 0) = T(motion_weight)*p_error[1];
        Residuals(2, 0) = T(motion_weight)*p_error[2];

        Residuals(3, 0) = T(motion_weight)*delta_q.x() * T(2);
        Residuals(4, 0) = T(motion_weight)*delta_q.y() * T(2);
        Residuals(5, 0) = T(motion_weight)*delta_q.z() * T(2);

        Residuals(6, 0) = T(motion_weight)*v_error[0];
        Residuals(7, 0) = T(motion_weight)*v_error[1];
        Residuals(8, 0) = T(motion_weight)*v_error[2];

        Residuals(9, 0) = T(motion_weight)*w_error[0]  * T(0.02);
        Residuals(10, 0) = T(motion_weight)*w_error[1] * T(0.02);
        Residuals(11, 0) = T(motion_weight)*w_error[2] * T(0.02);

        Residuals(12, 0) = T(motion_weight)*a_error[0] * T(0.1);
        Residuals(13, 0) = T(motion_weight)*a_error[1] * T(0.1);
        Residuals(14, 0) = T(motion_weight)*a_error[2] * T(0.1);


        for (int i = 0; i < 15; i++)
            residuals[i] = Residuals(i, 0);

        return true;
    }

    static ceres::CostFunction *Create(const double dt, Eigen::MatrixXd Cov_proir, double motion_weight)
    {
        return (new ceres::AutoDiffCostFunction<ConstAccFactorAuto, 15, 7, 3, 3, 3, 7, 3, 3, 3>(new ConstAccFactorAuto(dt, Cov_proir, motion_weight)));
    }

    double dt;
    Eigen::MatrixXd Cov_proir;
    double motion_weight;
};

struct CVRTFactorAuto
{
    CVRTFactorAuto(double _dt, double _motion_weight) : dt(_dt), motion_weight(_motion_weight) {}

    template <typename T>
    bool operator()(const T *const pose_pre, const T *const linearVel_pre, const T *const angularVel_pre,
                    const T *const pose_now, const T *const linearVel_now, const T *const angularVel_now, T *residuals) const
    {
        // Note: P:W系   V:W系   W:b系   A:b系
        Eigen::Matrix<T, 3, 1> P_pre(pose_pre[0], pose_pre[1], pose_pre[2]);
        Eigen::Matrix<T, 3, 1> V_pre(linearVel_pre[0], linearVel_pre[1], linearVel_pre[2]);
        Eigen::Matrix<T, 3, 1> W_pre(angularVel_pre[0], angularVel_pre[1], angularVel_pre[2]);
        Eigen::Quaternion<T> Q_pre{pose_pre[6], pose_pre[3], pose_pre[4], pose_pre[5]};
        Eigen::Matrix<T, 3, 3> R_pre = Q_pre.toRotationMatrix();

        Eigen::Matrix<T, 3, 1> P_now(pose_now[0], pose_now[1], pose_now[2]);
        Eigen::Matrix<T, 3, 1> V_now(linearVel_now[0], linearVel_now[1], linearVel_now[2]);
        Eigen::Matrix<T, 3, 1> W_now(angularVel_now[0], angularVel_now[1], angularVel_now[2]);
        Eigen::Quaternion<T> Q_now{pose_now[6], pose_now[3], pose_now[4], pose_now[5]};

        Eigen::Matrix<T, 3, 1> p_error = P_now - (P_pre + Q_pre * V_pre * dt);
        Eigen::Quaternion<T> delta_q = (Q_pre * Utility::deltaQ(W_pre * dt)).conjugate() * Q_now;
        Eigen::Matrix<T, 3, 1> v_error = V_now - V_pre;
        Eigen::Matrix<T, 3, 1> w_error = W_now - W_pre;
      
        residuals[0] = T(motion_weight) * p_error[0]* T(20);
        residuals[1] = T(motion_weight) * p_error[1]* T(20);
        residuals[2] = T(motion_weight) * p_error[2]* T(20);

        residuals[3] = T(motion_weight) * delta_q.x() * T(2);
        residuals[4] = T(motion_weight) * delta_q.y() * T(2);
        residuals[5] = T(motion_weight) * delta_q.z() * T(2);

        residuals[6] = T(motion_weight) * v_error[0];
        residuals[7] = T(motion_weight) * v_error[1];
        residuals[8] = T(motion_weight) * v_error[2];

        residuals[9]  = T(motion_weight) * w_error[0];
        residuals[10] = T(motion_weight) * w_error[1];
        residuals[11] = T(motion_weight) * w_error[2];


        return true;
    }

    static ceres::CostFunction *Create(const double dt, double motion_weight)
    {
        return (new ceres::AutoDiffCostFunction<CVRTFactorAuto, 12, 7, 3, 3, 7, 3, 3>(new CVRTFactorAuto(dt, motion_weight)));
    }

    double dt;
    double motion_weight;
};


struct MeasureFactor
{
    MeasureFactor(Eigen::Vector3d Pi_obs_, Eigen::Quaterniond Qi_obs_, double score_): 
            Pi_obs(Pi_obs_), Qi_obs(Qi_obs_), score(score_) {}
    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        // Note: P:W系   V:W系   W:b系   A:b系
        Eigen::Matrix<T, 3, 1> P_now(pose[0], pose[1], pose[2]);
        Eigen::Quaternion<T> Q_now{pose[6], pose[3], pose[4], pose[5]};

        Eigen::Matrix<T, 3, 1> P_obs((T)Pi_obs.x(), (T)Pi_obs.y(), (T)Pi_obs.z());
        Eigen::Quaternion<T> Q_obs{(T)Qi_obs.w(), (T)Qi_obs.x(), (T)Qi_obs.y(), (T)Qi_obs.z()};

        Eigen::Matrix<T, 3, 1> p_error = P_now - P_obs;
        Eigen::Quaternion<T> delta_q = Q_obs.conjugate() * Q_now;

        residuals[0] = T(score) * p_error[0];
        residuals[1] = T(score) * p_error[1];
        residuals[2] = T(score) * p_error[2];
        residuals[3] = T(score) * delta_q.x() * T(2);
        residuals[4] = T(score) * delta_q.y() * T(2);
        residuals[5] = T(score) * delta_q.z() * T(2);

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d Pi_obs_, Eigen::Quaterniond Qi_obs_, double score_)
    {
        return (new ceres::AutoDiffCostFunction<MeasureFactor, 6, 7>(new MeasureFactor(Pi_obs_, Qi_obs_, score_)));
    }

    Eigen::Vector3d Pi_obs;
    Eigen::Quaterniond Qi_obs;
    double score;
    bool is_static;
};


#endif // LIDARFACTOR_H
