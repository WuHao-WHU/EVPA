#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <ceres/ceres.h>


#define HASH_P 116101
#define MAX_N 10000000000
#define PLANE_FITTING_NUM 5

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointCloud<PointType>::Ptr PointCloudPtr;



struct trajectory
{
    double para_q[4];
    double para_t[3];
    trajectory():covariance_q_t({0}){
        para_q[0]=0;    //  x
        para_q[1]=0;    //  y
        para_q[2]=0;    //  z
        para_q[3]=1;    //  w

        para_t[0]=0;    //  x
        para_t[1]=0;    //  y
        para_t[2]=0;    //  z

        //covariance_q_t = {0};
        //covariance_t = {0};
    }

    trajectory(double x,double y,double z,double w,double t_x,double t_y,double t_z)
    {
        para_q[0]= x;
        para_q[1]= y;
        para_q[2]= z;
        para_q[3]= w;

        para_t[0]=t_x;
        para_t[1]=t_y;
        para_t[2]=t_z;
    }

    double covariance_q_t[36];
    //double covariance_t[9];
};

class tool
{
public:
    static void TransformPointCloud(PointCloudPtr &in,PointCloudPtr &out,trajectory &traj);
    static void TransformPoint(PointType &in,PointType &out, trajectory &traj);
    static int sampleLeafsized( pcl::PointCloud<PointType>::Ptr& cloud_in, pcl::PointCloud<PointType>& cloud_out, float downsample_size);
    //static void SetMarker(visualization_msgs::Marker &mark,visualization_msgs::Marker::CUBE)
};

class TicToc
{
public:
    TicToc()
    {

    }
    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count();
    }
private:
    std::chrono::time_point<std::chrono::system_clock> start,end;
};

class Point2PlaneOptimize : public ceres::SizedCostFunction<1, 4, 3>
{
public:
    Point2PlaneOptimize(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
                        double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                        negative_OA_dot_norm(negative_OA_dot_norm_) {}
    virtual ~Point2PlaneOptimize() {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        Eigen::Matrix<double, 3, 1> cp{(curr_point.x()), (curr_point.y()), (curr_point.z())};
        // Eigen::Matrix<double, 3, 1> lpa{(last_point_a.x()), (last_point_a.y()), (last_point_a.z())};
        // Eigen::Matrix<double, 3, 1> lpb{(last_point_b.x()), (last_point_b.y()), (last_point_b.z())};

        Eigen::Quaternion<double> q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

        Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;
        double distance = plane_unit_norm.dot(lp) + negative_OA_dot_norm;
        double s = 1 - 0.9 * fabs(distance) / sqrt(sqrt(lp.x() * lp.x() + lp.y() * lp.y() + lp.z() * lp.z()));
        // double s = 1;
        residuals[0] = s * distance;

        Eigen::Matrix<double, 1, 3> Jacobian_X;
        Jacobian_X << s * plane_unit_norm.x(), s * plane_unit_norm.y(), s * plane_unit_norm.z();

        // 对四元数求导数
        double qwX = q_last_curr.w() * cp.x();
        double qwY = q_last_curr.w() * cp.y();
        double qwZ = q_last_curr.w() * cp.z();

        double qxX = q_last_curr.x() * cp.x();
        double qxY = q_last_curr.x() * cp.y();
        double qxZ = q_last_curr.x() * cp.z();

        double qyX = q_last_curr.y() * cp.x();
        double qyY = q_last_curr.y() * cp.y();
        double qyZ = q_last_curr.y() * cp.z();

        double qzX = q_last_curr.z() * cp.x();
        double qzY = q_last_curr.z() * cp.y();
        double qzZ = q_last_curr.z() * cp.z();
        Eigen::Matrix<double, 3, 4> Jacobian_Q;
        // //Quaternion
        // Jacobian_Q<<qyZ-qzY,qyY+qzZ, qwZ+qxY-2*qyX, -qwY+qxZ-2*qzX,
        //                             -qxZ+qzX, -qwZ-2*qxY+qyX, qxX+qzZ, qwX+qyZ-2*qzY,
        //                             qxY-qyX, qwY-2*qxZ+qzX, -qwX-2*qyZ+qzY, qxX+qyY;
        // EigenQuaternion
        Jacobian_Q << qyY + qzZ, qwZ + qxY - 2 * qyX, -qwY + qxZ - 2 * qzX, qyZ - qzY,
            -qwZ - 2 * qxY + qyX, qxX + qzZ, qwX + qyZ - 2 * qzY, -qxZ + qzX,
            qwY - 2 * qxZ + qzX, -qwX - 2 * qyZ + qzY, qxX + qyY, qxY - qyX;
        Eigen::Matrix<double, 1, 4> Jacobian_total = Jacobian_X * Jacobian_Q;
        if (jacobians != NULL)
        {
            if (jacobians[0] != NULL)
            {
                jacobians[0][0] = Jacobian_total[0];
                jacobians[0][1] = Jacobian_total[1];
                jacobians[0][2] = Jacobian_total[2];
                jacobians[0][3] = Jacobian_total[3];
            }
            if (jacobians[1] != NULL)
            {
                jacobians[1][0] = Jacobian_X[0];
                jacobians[1][1] = Jacobian_X[1];
                jacobians[1][2] = Jacobian_X[2];
            }
        }

        return true;
    }

private:
    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Point2PointAuto
{
public:
    Point2PointAuto(Eigen::Vector3d curr,Eigen::Vector3d map):curr_point(curr),map_point(map){}
    template <typename T>
    bool operator()(
        const T* const para_q,
        const T* const para_t,
        T* residuals_ptr
    ) const
    {
        Eigen::Quaternion<T> q_last_curr(para_q[3],para_q[0],para_q[1],para_q[2]);
        Eigen::Matrix<T,3,1> t_last_curr(para_t[0],para_t[1],para_t[2]);

        Eigen::Matrix<T,3,1> map_estimate = q_last_curr* curr_point.template cast<T>()+t_last_curr;

        T distance = (map_estimate-map_point.template cast<T>()).norm();

        residuals_ptr[0] = distance;

        // Eigen::Matrix<T,1,3> Jacobian_X;
        // Jacobian_X << ((map_estimate).x()-map_point.x())/distance,((map_estimate).y()-map_point.y())/distance,((map_estimate).z()-map_point.z())/distance;

        // double qwX = q_last_curr.w() * curr_point.x();
        // double qwY = q_last_curr.w() * curr_point.y();
        // double qwZ = q_last_curr.w() * curr_point.z();

        // double qxX = q_last_curr.x() * curr_point.x();
        // double qxY = q_last_curr.x() * curr_point.y();
        // double qxZ = q_last_curr.x() * curr_point.z();

        // double qyX = q_last_curr.y() * curr_point.x();
        // double qyY = q_last_curr.y() * curr_point.y();
        // double qyZ = q_last_curr.y() * curr_point.z();

        // double qzX = q_last_curr.z() * curr_point.x();
        // double qzY = q_last_curr.z() * curr_point.y();
        // double qzZ = q_last_curr.z() * curr_point.z();
        // Eigen::Matrix<double, 3, 4> Jacobian_Q;
        // // //Quaternion
        // // Jacobian_Q<<qyZ-qzY,qyY+qzZ, qwZ+qxY-2*qyX, -qwY+qxZ-2*qzX,
        // //                             -qxZ+qzX, -qwZ-2*qxY+qyX, qxX+qzZ, qwX+qyZ-2*qzY,
        // //                             qxY-qyX, qwY-2*qxZ+qzX, -qwX-2*qyZ+qzY, qxX+qyY;
        // // EigenQuaternion
        // Jacobian_Q << qyY + qzZ, qwZ + qxY - 2 * qyX, -qwY + qxZ - 2 * qzX, qyZ - qzY,
        //     -qwZ - 2 * qxY + qyX, qxX + qzZ, qwX + qyZ - 2 * qzY, -qxZ + qzX,
        //     qwY - 2 * qxZ + qzX, -qwX - 2 * qyZ + qzY, qxX + qyY, qxY - qyX;

        // Eigen::Matrix<double,1,4> Jacobian_total = Jacobian_X*Jacobian_Q;

        // if (jacobians != NULL)
        // {
        //     if (jacobians[0] != NULL)
        //     {
        //         jacobians[0][0] = Jacobian_total[0];
        //         jacobians[0][1] = Jacobian_total[1];
        //         jacobians[0][2] = Jacobian_total[2];
        //         jacobians[0][3] = Jacobian_total[3];
        //     }
        //     if (jacobians[1] != NULL)
        //     {
        //         jacobians[1][0] = 1;
        //         jacobians[1][1] = 1;
        //         jacobians[1][2] = 1;
        //     }
        // }

        return true;
    }
    static ceres::CostFunction* Create(
        Eigen::Vector3d curr,
        Eigen::Vector3d t_m_d
    )
    {
        return new ceres::AutoDiffCostFunction<Point2PointAuto,1,4,3>(new Point2PointAuto(curr,t_m_d));
    }

private:
    Eigen::Vector3d curr_point;
    Eigen::Vector3d map_point;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class Point2PointOptimize:public ceres::SizedCostFunction<1,4,3>
{
public:
    Point2PointOptimize(Eigen::Vector3d curr,Eigen::Vector3d map):curr_point(curr),map_point(map){}
    virtual ~Point2PointOptimize(){}

    virtual bool Evaluate(
        double const *const* parameters,
        double *residuals,
        double **jacobians
    )const
    {
        Eigen::Quaterniond q_last_curr(parameters[0][3],parameters[0][0],parameters[0][1],parameters[0][2]);
        Eigen::Vector3d t_last_curr(parameters[1][0],parameters[1][1],parameters[1][2]);

        Eigen::Vector3d map_estimate = q_last_curr* curr_point+t_last_curr;

        double distance = (map_estimate-map_point).norm();

        residuals[0] = distance;

        Eigen::Matrix<double,1,3> Jacobian_X;
        Jacobian_X << ((map_estimate).x()-map_point.x())/distance,((map_estimate).y()-map_point.y())/distance,((map_estimate).z()-map_point.z())/distance;

        double qwX = q_last_curr.w() * curr_point.x();
        double qwY = q_last_curr.w() * curr_point.y();
        double qwZ = q_last_curr.w() * curr_point.z();

        double qxX = q_last_curr.x() * curr_point.x();
        double qxY = q_last_curr.x() * curr_point.y();
        double qxZ = q_last_curr.x() * curr_point.z();

        double qyX = q_last_curr.y() * curr_point.x();
        double qyY = q_last_curr.y() * curr_point.y();
        double qyZ = q_last_curr.y() * curr_point.z();

        double qzX = q_last_curr.z() * curr_point.x();
        double qzY = q_last_curr.z() * curr_point.y();
        double qzZ = q_last_curr.z() * curr_point.z();
        Eigen::Matrix<double, 3, 4> Jacobian_Q;
        // //Quaternion
        // Jacobian_Q<<qyZ-qzY,qyY+qzZ, qwZ+qxY-2*qyX, -qwY+qxZ-2*qzX,
        //                             -qxZ+qzX, -qwZ-2*qxY+qyX, qxX+qzZ, qwX+qyZ-2*qzY,
        //                             qxY-qyX, qwY-2*qxZ+qzX, -qwX-2*qyZ+qzY, qxX+qyY;
        // EigenQuaternion
        Jacobian_Q << qyY + qzZ, qwZ + qxY - 2 * qyX, -qwY + qxZ - 2 * qzX, qyZ - qzY,
            -qwZ - 2 * qxY + qyX, qxX + qzZ, qwX + qyZ - 2 * qzY, -qxZ + qzX,
            qwY - 2 * qxZ + qzX, -qwX - 2 * qyZ + qzY, qxX + qyY, qxY - qyX;

        Eigen::Matrix<double,1,4> Jacobian_total = Jacobian_X*Jacobian_Q;

        if (jacobians != NULL)
        {
            if (jacobians[0] != NULL)
            {
                jacobians[0][0] = Jacobian_total[0];
                jacobians[0][1] = Jacobian_total[1];
                jacobians[0][2] = Jacobian_total[2];
                jacobians[0][3] = Jacobian_total[3];
            }
            if (jacobians[1] != NULL)
            {
                jacobians[1][0] = 1;
                jacobians[1][1] = 1;
                jacobians[1][2] = 1;
            }
        }

        return true;
    }
private:
    Eigen::Vector3d curr_point;
    Eigen::Vector3d map_point;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


class PoseGraphOptimize:public ceres::SizedCostFunction<6,4,3,4,3>
{
public:
    PoseGraphOptimize(double* q_m_d,double* t_m_d)
    {  
        q_m.x() = q_m_d[0];
        q_m.y() = q_m_d[1];
        q_m.z() = q_m_d[2];
        q_m.w() = q_m_d[3];

        t_m.x() = t_m_d[0];
        t_m.y() = t_m_d[1];
        t_m.z() = t_m_d[2];
    }

    virtual ~PoseGraphOptimize(){}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        Eigen::Quaterniond q_a(parameters[0][3],parameters[0][0],parameters[0][1],parameters[0][2]);
        Eigen::Vector3d t_a(parameters[1][0],parameters[1][1],parameters[1][2]);
        Eigen::Quaterniond q_b(parameters[2][3],parameters[2][0],parameters[2][1],parameters[2][2]);
        Eigen::Vector3d t_b(parameters[3][0],parameters[3][1],parameters[3][2]);

        Eigen::Quaterniond q_a_inv = q_a.conjugate();
        Eigen::Quaterniond q_estimate = q_a_inv*q_b;

        Eigen::Vector3d t_estimate = q_a_inv*(t_b-t_a);

        Eigen::Quaterniond delta_q = q_m.conjugate()*q_estimate;

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(residuals);

        residuals_map.block<3,1>(0,0) = t_estimate - t_m;
        residuals_map.block<3,1>(3,0) = 2* delta_q.vec();

        Eigen::Matrix<double, 6, 4> Jacobian_q_a;
        Eigen::Matrix<double, 6, 3> Jacobian_t_a;
        Eigen::Matrix<double, 6, 4> Jacobian_q_b;
        Eigen::Matrix<double, 6, 3> Jacobian_t_b;

        double q_m_x=q_m.x();
        double q_m_y=q_m.y();
        double q_m_z=q_m.z();
        double q_m_w=q_m.w();

        double q_b_x=q_b.x();
        double q_b_y=q_b.y();
        double q_b_z=q_b.z();
        double q_b_w=q_b.w();

        double q_a_x=q_a.x();
        double q_a_y=q_a.y();
        double q_a_z=q_a.z();
        double q_a_w=q_a.w();

        double t_b_x=t_b.x();
        double t_b_y=t_b.y();
        double t_b_z=t_b.z();

        double t_a_x=t_a.x();
        double t_a_y=t_a.y();
        double t_a_z=t_a.z();

        Jacobian_q_a.block<3,4>(0,0)<<
        -q_a_x*(t_a_x - t_b_x) - q_a_y*(t_a_y - t_b_y) - q_a_z*(t_a_z - t_b_z),q_a_y*(t_a_x - t_b_x) - q_a_x*(t_a_y - t_b_y) + q_a_w*(t_a_z - t_b_z),q_a_z*(t_a_x - t_b_x) - q_a_w*(t_a_y - t_b_y) - q_a_x*(t_a_z - t_b_z),q_a_y*(t_a_z - t_b_z) - q_a_z*(t_a_y - t_b_y) - q_a_w*(t_a_x - t_b_x),
        q_a_x*(t_a_y - t_b_y) - q_a_y*(t_a_x - t_b_x) - q_a_w*(t_a_z - t_b_z),-q_a_x*(t_a_x - t_b_x) - q_a_y*(t_a_y - t_b_y) - q_a_z*(t_a_z - t_b_z),q_a_w*(t_a_x - t_b_x) + q_a_z*(t_a_y - t_b_y) - q_a_y*(t_a_z - t_b_z),q_a_z*(t_a_x - t_b_x) - q_a_w*(t_a_y - t_b_y) - q_a_x*(t_a_z - t_b_z),
        q_a_w*(t_a_y - t_b_y) - q_a_z*(t_a_x - t_b_x) + q_a_x*(t_a_z - t_b_z),q_a_y*(t_a_z - t_b_z) - q_a_z*(t_a_y - t_b_y) - q_a_w*(t_a_x - t_b_x),- q_a_x*(t_a_x - t_b_x) - q_a_y*(t_a_y - t_b_y) - q_a_z*(t_a_z - t_b_z),q_a_x*(t_a_y - t_b_y) - q_a_y*(t_a_x - t_b_x) - q_a_w*(t_a_z - t_b_z);

        Jacobian_q_a.block<3,4>(3,0)<<
        q_b_y*q_m_y - q_b_x*q_m_x - q_b_w*q_m_w + q_b_z*q_m_z,- q_b_w*q_m_z - q_b_x*q_m_y - q_b_y*q_m_x - q_b_z*q_m_w,q_b_w*q_m_y + q_b_y*q_m_w - q_b_x*q_m_z - q_b_z*q_m_x,q_b_x*q_m_w - q_b_w*q_m_x + q_b_y*q_m_z - q_b_z*q_m_y,
        q_b_w*q_m_z - q_b_x*q_m_y - q_b_y*q_m_x + q_b_z*q_m_w,q_b_x*q_m_x - q_b_w*q_m_w - q_b_y*q_m_y + q_b_z*q_m_z,- q_b_w*q_m_x - q_b_x*q_m_w - q_b_y*q_m_z - q_b_z*q_m_y,q_b_y*q_m_w - q_b_w*q_m_y - q_b_x*q_m_z + q_b_z*q_m_x,
        - q_b_w*q_m_y - q_b_y*q_m_w - q_b_x*q_m_z - q_b_z*q_m_x,q_b_w*q_m_x + q_b_x*q_m_w - q_b_y*q_m_z - q_b_z*q_m_y,q_b_x*q_m_x - q_b_w*q_m_w + q_b_y*q_m_y - q_b_z*q_m_z,q_b_x*q_m_y - q_b_w*q_m_z - q_b_y*q_m_x + q_b_z*q_m_w;

        Jacobian_t_a.block<3,3>(0,0)<<
        - q_a_w*q_a_w - q_a_x*q_a_x + q_a_y*q_a_y + q_a_z*q_a_z,- 2*q_a_w*q_a_z - 2*q_a_x*q_a_y,2*q_a_w*q_a_y - 2*q_a_x*q_a_z,
        2*q_a_w*q_a_z - 2*q_a_x*q_a_y,- q_a_w*q_a_w + q_a_x*q_a_x - q_a_y*q_a_y + q_a_z*q_a_z,- 2*q_a_w*q_a_x - 2*q_a_y*q_a_z,
        - 2*q_a_w*q_a_y - 2*q_a_x*q_a_z,2*q_a_w*q_a_x - 2*q_a_y*q_a_z, - q_a_w*q_a_w + q_a_x*q_a_x+ q_a_y*q_a_y - q_a_z*q_a_z;
        Jacobian_t_a.block<3,3>(3,0).setZero();

        Jacobian_q_b.block<3,4>(0,0).setZero();
        Jacobian_q_b.block<3,4>(3,0)<<
        //q_a_w*q_m_x + q_a_x*q_m_w + q_a_y*q_m_z - q_a_z*q_m_y, q_a_w*q_m_y + q_a_y*q_m_w - q_a_x*q_m_z + q_a_z*q_m_x, q_a_w*q_m_z + q_a_x*q_m_y - q_a_y*q_m_x + q_a_z*q_m_w,q_a_w*q_m_w - q_a_x*q_m_x - q_a_y*q_m_y - q_a_z*q_m_z,
        q_a_w*q_m_w - q_a_x*q_m_x - q_a_y*q_m_y - q_a_z*q_m_z, q_a_w*q_m_z + q_a_x*q_m_y - q_a_y*q_m_x + q_a_z*q_m_w, q_a_x*q_m_z - q_a_y*q_m_w - q_a_w*q_m_y - q_a_z*q_m_x,q_a_z*q_m_y - q_a_x*q_m_w - q_a_y*q_m_z - q_a_w*q_m_x,
        q_a_y*q_m_x - q_a_x*q_m_y - q_a_w*q_m_z - q_a_z*q_m_w, q_a_w*q_m_w - q_a_x*q_m_x - q_a_y*q_m_y - q_a_z*q_m_z, q_a_w*q_m_x + q_a_x*q_m_w + q_a_y*q_m_z - q_a_z*q_m_y,q_a_x*q_m_z - q_a_y*q_m_w - q_a_w*q_m_y - q_a_z*q_m_x, 
        q_a_w*q_m_y + q_a_y*q_m_w - q_a_x*q_m_z + q_a_z*q_m_x, q_a_z*q_m_y - q_a_x*q_m_w - q_a_y*q_m_z - q_a_w*q_m_x, q_a_w*q_m_w - q_a_x*q_m_x - q_a_y*q_m_y - q_a_z*q_m_z,q_a_y*q_m_x - q_a_x*q_m_y - q_a_w*q_m_z - q_a_z*q_m_w;
        // q_a_w*q_m_w - q_a_x*q_m_x - q_a_y*q_m_y - q_a_z*q_m_z,q_a_w*q_m_z + q_a_x*q_m_y - q_a_y*q_m_x + q_a_z*q_m_w,q_a_x*q_m_z - q_a_y*q_m_w - q_a_w*q_m_y - q_a_z*q_m_x,q_a_z*q_m_y - q_a_x*q_m_w - q_a_y*q_m_z - q_a_w*q_m_x,
        // q_a_y*q_m_x - q_a_x*q_m_y - q_a_w*q_m_z - q_a_z*q_m_w,q_a_w*q_m_w - q_a_x*q_m_x - q_a_y*q_m_y - q_a_z*q_m_z,q_a_w*q_m_x + q_a_x*q_m_w + q_a_y*q_m_z - q_a_z*q_m_y,q_a_x*q_m_z - q_a_y*q_m_w - q_a_w*q_m_y - q_a_z*q_m_x,
        // q_a_w*q_m_y + q_a_y*q_m_w - q_a_x*q_m_z + q_a_z*q_m_x,q_a_z*q_m_y - q_a_x*q_m_w - q_a_y*q_m_z - q_a_w*q_m_x,q_a_w*q_m_w - q_a_x*q_m_x - q_a_y*q_m_y - q_a_z*q_m_z,q_a_y*q_m_x - q_a_x*q_m_y - q_a_w*q_m_z - q_a_z*q_m_w;



        Jacobian_t_b.block<3,3>(0,0)<< 
        q_a_w*q_a_w + q_a_x*q_a_x - q_a_y*q_a_y - q_a_z*q_a_z, 2*q_a_w*q_a_z + 2*q_a_x*q_a_y,-2*q_a_w*q_a_y + 2*q_a_x*q_a_z,
        -2*q_a_w*q_a_z + 2*q_a_x*q_a_y, q_a_w*q_a_w - q_a_x*q_a_x + q_a_y*q_a_y - q_a_z*q_a_z, 2*q_a_w*q_a_x + 2*q_a_y*q_a_z,
         2*q_a_w*q_a_y + 2*q_a_x*q_a_z,-2*q_a_w*q_a_x + 2*q_a_y*q_a_z,  q_a_w*q_a_w - q_a_x*q_a_x- q_a_y*q_a_y + q_a_z*q_a_z;
         Jacobian_t_b.block<3,3>(3,0).setZero();
        if (jacobians != NULL)
        {
            //q_a
            if (jacobians[0] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double,6,4,Eigen::RowMajor>> jacobian(jacobians[0]);
               
                jacobian=Jacobian_q_a;

            }
            //t_a
            if (jacobians[1] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> jacobian(jacobians[1]);
                jacobian = Jacobian_t_a;
            }
            //q_b
            if (jacobians[2] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double,6,4,Eigen::RowMajor>> jacobian(jacobians[2]);
                jacobian = Jacobian_q_b;
            }
            //t_b
            if (jacobians[3] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> jacobian(jacobians[3]);
                jacobian =Jacobian_t_b;
            }
        }
        return true;
    }

private:
    Eigen::Quaternion<double> q_m;
    Eigen::Vector3d t_m;
};


struct PoseGraphOptimizeAuto
{
    // PoseGraphOptimizeAuto(const double* q_m_d,const double* t_m_d,double* cov):cov_{cov}
    // {
    //     q_m.x() = q_m_d[0];
    //     q_m.y() = q_m_d[1];
    //     q_m.z() = q_m_d[2];
    //     q_m.w() = q_m_d[3];

    //     t_m.x() = t_m_d[0];
    //     t_m.y() = t_m_d[1];
    //     t_m.z() = t_m_d[2];
    // }

    PoseGraphOptimizeAuto(const double* q_m_d,const double* t_m_d)
    {
        q_m.x() = q_m_d[0];
        q_m.y() = q_m_d[1];
        q_m.z() = q_m_d[2];
        q_m.w() = q_m_d[3];

        t_m.x() = t_m_d[0];
        t_m.y() = t_m_d[1];
        t_m.z() = t_m_d[2];
    }

    template <typename T>
    bool operator()(
        const T* const q_a_ptr,
        const T* const t_a_ptr,
        const T* const q_b_ptr,
        const T* const t_b_ptr,
        T* residuals_ptr
    ) const
    {
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_a(t_a_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);

        Eigen::Map<const Eigen::Matrix<T,3,1>> t_b(t_b_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

        //Eigen::Matrix<double,6,6> cov_temp = cov_.llt().matrixL();

        Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
        Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;


        Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (t_b - t_a);

        Eigen::Quaternion<T> delta_q =
        q_m.template cast<T>() * q_ab_estimated.conjugate();

        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
        
        // Compute the residuals.
        // [ position         ]   [ delta_p          ]
        // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
        residuals.template block<3, 1>(0, 0) = p_ab_estimated - t_m.template cast<T>();
        residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

        //residuals.applyOnTheLeft(cov_temp.template cast<T>());

        return true;
    }

    // static ceres::CostFunction* Create(
    //     const double* q_m_d,
    //     const double* t_m_d,
    //     double* cov
    // )
    // {
    //     return new ceres::AutoDiffCostFunction<PoseGraphOptimizeAuto,6,4,3,4,3>(new PoseGraphOptimizeAuto(q_m_d,t_m_d,cov));
    // }

    static ceres::CostFunction* Create(
        const double* q_m_d,
        const double* t_m_d
    )
    {
        return new ceres::AutoDiffCostFunction<PoseGraphOptimizeAuto,6,4,3,4,3>(new PoseGraphOptimizeAuto(q_m_d,t_m_d));
    }
    
private:
    Eigen::Quaternion<double> q_m;
    Eigen::Vector3d t_m;  
    //Eigen::Map<Eigen::Matrix<double,6,6,Eigen::RowMajor>> cov_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif