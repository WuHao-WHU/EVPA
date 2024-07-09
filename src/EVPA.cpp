#include "EVPA.h"

void EVPA::DisplayVoxel(ros::Publisher &pub_voxel,ros::Publisher &pub_plane,ros::Publisher &pub_notplane)
{

}

void EVPA::CutVoxel(PointCloudPtr &pcd,trajectory &traj,int n_scan)
{
    float voxel_location[3];
    for(int i =0;i<pcd->points.size();i++)
    {
        PointType p_t;
        tool::TransformPoint(pcd->points[i],p_t,traj);
        Eigen::Vector3d p_t_e(p_t.x,p_t.y,p_t.z);
        Eigen::Vector3d p_o_e(pcd->points[i].x,pcd->points[i].y,pcd->points[i].z);
        voxel_location[0] = p_t.x/voxel_size;
        voxel_location[1] = p_t.y/voxel_size;
        voxel_location[2] = p_t.z/voxel_size;
        if(voxel_location[0]<0) voxel_location[0] -=1.0;
        if(voxel_location[1]<0) voxel_location[1] -=1.0;
        if(voxel_location[2]<0) voxel_location[2] -=1.0;
        
        VoxelLocation loc( (int64_t)voxel_location[0],(int64_t)voxel_location[1],(int64_t)voxel_location[2]);

        std::unordered_map<VoxelLocation,Voxel*>::iterator iter = Voxelized_map.find(loc);

        if(iter!=Voxelized_map.end())
        {
            iter->second->PushBack(p_o_e,p_t_e,n_scan);
        }
        else
        {
            Voxel* vox =new Voxel(pose_size);
            vox->PushBack(p_o_e,p_t_e,n_scan);
            
            vox->voxel_center[0] = (0.5+loc.x)*voxel_size;
            vox->voxel_center[1] = (0.5+loc.y)*voxel_size;
            vox->voxel_center[2] = (0.5+loc.z)*voxel_size;

            Voxelized_map[loc]=vox;
        }

    }
}

void EVPA::SolveWithFactorGraph(ros::Publisher &pub_voxel,ros::Publisher &pub_plane,ros::Publisher &pub_notplane,std::vector<trajectory> &traj_s)
{
    LOG(INFO)<<"begin to construct factor graph...\n";
    TicToc t_solver;
    t_solver.tic();
    //std::unordered_map<int,int> problem_indices;
    std::vector<int> problem_indices;
    std::unordered_map<int,ceres::Problem*> problems;
    //ceres::Problem problem;


    last_plane_count = plane_count;
    overlap_count = 0;
    plane_count = 0;
    notplane_count = 0;
    //PointCloudPtr plane(new PointCloud);
    //PointCloudPtr notplane(new PointCloud);

#pragma omp parallel
    {
#pragma omp single
        {
            for (std::unordered_map<VoxelLocation, Voxel *>::iterator iter = Voxelized_map.begin(); iter != Voxelized_map.end(); iter++)
#pragma omp task
            {

                // 寻找重叠部分的体素
                std::vector<int> pose_index;
                // std::unordered_map<int,std::pair<int,int>> relative_pose_graph_this_voxel;

                for (int i = 0; i < pose_size; i++)
                {
                    // 30
                    if (iter->second->pts_trans[i].size() < 15)
                        continue;

                    pose_index.push_back(i);
                }
                
                // if (pose_index.size() < 2)
                //     continue;

                //
#pragma omp critical
                {
                    if (pose_index.size() >= 3)
                        overlap_count++;
                }

                // plane factors
                if (pose_index.size() >= 2)
                {

                    bool judge = iter->second->JudgePlane(pose_size);
                    // 如果有重叠部分的体素，则构建因子图,从后往前（root）配准
                    // i -> j
                    for (int i = pose_index.size() - 1; i >= 1; i--)
                    {
                        for (int j = i - 1; j >= 0; j--)
                        {
                            if(pose_index[i]-pose_index[j]>=reliable_region)
                                break;
                            int relative_index = pose_index[i] + pose_index[j] * pose_size;
#pragma omp critical
                            {
                                if (relative_pose_graph.find(relative_index) == relative_pose_graph.end())
                                {
                                    // ceres::Problem* problem = new ceres::Problem;
                                    ceres::Problem *problem = new ceres::Problem;
                                    ceres::Manifold *manifold = new ceres::EigenQuaternionManifold();
                                    //problem_indices.push_back(relative_index);
                                    problems[relative_index] = problem;
                                    // problem_indices[relative_index] = problems.size()-1;

                                    relative_pose_graph[relative_index] = std::make_pair(pose_index[i], pose_index[j]);

                                    Eigen::Quaterniond q_j(traj_s[pose_index[j]].para_q[3], traj_s[pose_index[j]].para_q[0], traj_s[pose_index[j]].para_q[1], traj_s[pose_index[j]].para_q[2]);
                                    Eigen::Quaterniond q_i(traj_s[pose_index[i]].para_q[3], traj_s[pose_index[i]].para_q[0], traj_s[pose_index[i]].para_q[1], traj_s[pose_index[i]].para_q[2]);

                                    Eigen::Vector3d t_i(traj_s[pose_index[i]].para_t[0], traj_s[pose_index[i]].para_t[1], traj_s[pose_index[i]].para_t[2]);
                                    Eigen::Vector3d t_j(traj_s[pose_index[j]].para_t[0], traj_s[pose_index[j]].para_t[1], traj_s[pose_index[j]].para_t[2]);

                                    Eigen::Quaterniond q_j_i = q_j.conjugate() * q_i;
                                    Eigen::Vector3d t_j_i = q_j.conjugate() * (t_i - t_j);

                                    trajectory temp(q_j_i.x(), q_j_i.y(), q_j_i.z(), q_j_i.w(), t_j_i.x(), t_j_i.y(), t_j_i.z());
                                    // std::cout<<"original "<<relative_index<<"-> \n"<<q_j_i.toRotationMatrix()<<"\n";
                                    // std::cout<<t_j_i<<"\n";
                                    relative_pose[relative_index] = temp;
                                    relative_pose_point[relative_index] = temp;
                                    relative_pose_point2point[relative_index] =temp;

                                    problems[relative_index]->AddParameterBlock(relative_pose[relative_index].para_q, 4, manifold);
                                    problems[relative_index]->AddParameterBlock(relative_pose[relative_index].para_t, 3);
                                    
                                    //problems[relative_index]->AddParameterBlock(relative_pose_point[relative_index].para_q, 4, manifold);
                                    //problems[relative_index]->AddParameterBlock(relative_pose_point[relative_index].para_t, 3);
                                    //problems[relative_index]->AddParameterBlock(relative_pose_point[relative_index].para_t, 3);

                                    // problems[relative_index].AddParameterBlock(relative_pose[relative_index].para_q, 4, manifold);
                                    // problems[relative_index].AddParameterBlock(relative_pose[relative_index].para_t, 3);
                                    //  problem.AddParameterBlock(relative_pose[pose_index[i] + pose_index[j] * pose_size].para_q, 4, manifold);
                                    //  problem.AddParameterBlock(relative_pose[pose_index[i] + pose_index[j] * pose_size].para_t, 3);
                                }
                            }
                            // add factors
                            if (judge)
                            {
#pragma omp critical
                                {
                                    plane_count++;
                                    Voxelized_map_plane_visualize.push_back(iter->second);
                                    pose_index_each_voxel.push_back(pose_index);
                                }
                                // 从大序号向小序号配准
                                // for (int i = pose_index.size() - 1; i >= 1; i--)
                                // {
                                //     for (int j = i - 1; j >= 0; j--)
                                //     {
                                // int relative_index = index_i + index_j * pose_size;
                                // relative_pose_graph_this_voxel[pose_index[i] + pose_index[j] * pose_size] = std::make_pair(pose_index[i], pose_index[j]);

                                // if(relative_pose.find(relative_index)== relative_pose.end())
                                // {
                                //     Eigen::Quaterniond q_j(traj_s[pose_index[j]].para_q[3], traj_s[pose_index[j]].para_q[0], traj_s[pose_index[j]].para_q[1], traj_s[pose_index[j]].para_q[2]);
                                //     Eigen::Quaterniond q_i(traj_s[pose_index[i]].para_q[3], traj_s[pose_index[i]].para_q[0], traj_s[pose_index[i]].para_q[1], traj_s[pose_index[i]].para_q[2]);

                                //     Eigen::Vector3d t_i(traj_s[pose_index[i]].para_t[0], traj_s[pose_index[i]].para_t[1], traj_s[pose_index[i]].para_t[2]);
                                //     Eigen::Vector3d t_j(traj_s[pose_index[j]].para_t[0], traj_s[pose_index[j]].para_t[1], traj_s[pose_index[j]].para_t[2]);

                                //     Eigen::Quaterniond q_j_i = q_j.conjugate() * q_i;
                                //     Eigen::Vector3d t_j_i = q_j.conjugate() * (t_i - t_j);

                                //     trajectory temp(q_j_i.x(), q_j_i.y(), q_j_i.z(), q_j_i.w(), t_j_i.x(), t_j_i.y(), t_j_i.z());
                                //     relative_pose[relative_index] = temp;
                                // }

                                PointCloudPtr target(new PointCloud);
                                for (int k = 0; k < iter->second->pts_orig[pose_index[j]].size(); k++)
                                {
                                    PointType temp(iter->second->pts_orig[pose_index[j]][k].x(), iter->second->pts_orig[pose_index[j]][k].y(), iter->second->pts_orig[pose_index[j]][k].z());
                                    target->points.push_back(temp);
                                }
                                pcl::KdTreeFLANN<PointType> kdtree;
                                kdtree.setInputCloud(target);

                                for (int k = 0; k < iter->second->pts_orig[pose_index[i]].size(); k++)
                                {
                                    std::vector<float> distances;
                                    std::vector<int> indices;
                                    PointType temp(iter->second->pts_orig[pose_index[i]][k].x(), iter->second->pts_orig[pose_index[i]][k].y(), iter->second->pts_orig[pose_index[i]][k].z());
                                    PointType temp_trans;
                                    tool::TransformPoint(temp, temp_trans, relative_pose[relative_index]);
                                    kdtree.nearestKSearch(temp_trans, PLANE_FITTING_NUM, indices, distances);
                                    Eigen::Matrix<double, PLANE_FITTING_NUM, 3> matA0;
                                    Eigen::Matrix<double, PLANE_FITTING_NUM, 1> matB0 = -1 * Eigen::Matrix<double, PLANE_FITTING_NUM, 1>::Ones();
                                    if (distances[4] < 1.0)
                                    {
                                        for (int j = 0; j < PLANE_FITTING_NUM; j++)
                                        {
                                            matA0(j, 0) = target->points[indices[j]].x;
                                            matA0(j, 1) = target->points[indices[j]].y;
                                            matA0(j, 2) = target->points[indices[j]].z;
                                        }
                                        // 计算平面法向量
                                        Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                                        double negative_OA_dot_norm = 1 / norm.norm();
                                        norm.normalize();

                                        // 如果有点距离平面太远，则该平面拟合的不好
                                        bool plane_valid = true;
                                        for (int j = 0; j < PLANE_FITTING_NUM; j++)
                                        {
                                            if (fabs(norm(0) * target->points[indices[j]].x +
                                                     norm(1) * target->points[indices[j]].y +
                                                     norm(2) * target->points[indices[j]].z + negative_OA_dot_norm) > 0.2)
                                            {
                                                plane_valid = false;
                                                break;
                                            }
                                        }
                                        Eigen::Vector3d curr_point(iter->second->pts_orig[pose_index[i]][k].x(), iter->second->pts_orig[pose_index[i]][k].y(), iter->second->pts_orig[pose_index[i]][k].z());
                                        if (plane_valid)
                                        {
                                            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                                            ceres::CostFunction *cost_function = new Point2PlaneOptimize(curr_point, norm, negative_OA_dot_norm);
#pragma omp critical
                                            {
                                                problems[relative_index]->AddResidualBlock(cost_function, loss_function, relative_pose[relative_index].para_q, relative_pose[relative_index].para_t);
                                                // problems[relative_index].AddResidualBlock(cost_function, nullptr, relative_pose[relative_index].para_q, relative_pose[relative_index].para_t);

                                                // problem.AddResidualBlock(cost_function, nullptr, relative_pose[pose_index[i] + pose_index[j] * pose_size].para_q, relative_pose[pose_index[i] + pose_index[j] * pose_size].para_t);
                                            }
                                        }
                                    }
                                }
                                //     }
                                // }
                            }
                            else
                            {
#pragma omp critical
                                {
                                    notplane_count++;
                                }

                                // for (int i = pose_index.size() - 1; i >= 1; i--)
                                // {
                                //     for (int j = i - 1; j >= 0; j--)
                                //     {
                                // int relative_index = pose_index[i] + pose_index[j] * pose_size;
                                // relative_pose_graph_this_voxel[pose_index[i] + pose_index[j] * pose_size] = std::make_pair(pose_index[i], pose_index[j]);
                                // #pragma omp critical
                                //                                 {
                                //                                     if (relative_pose_graph.find(relative_index) == relative_pose_graph.end())
                                //                                     {
                                //                                          //ceres::Problem* problem =new ceres::Problem;
                                //                                         ceres::Problem* problem = new ceres::Problem;
                                //                                         ceres::Manifold *manifold = new ceres::EigenQuaternionManifold();
                                //                                         problems[relative_index] = problem;
                                //                                         //problem_indices[relative_index] = problems.size()-1;

                                //                                         relative_pose_graph[relative_index] = std::make_pair(pose_index[i], pose_index[j]);

                                //                                         Eigen::Quaterniond q_j(traj_s[pose_index[j]].para_q[3], traj_s[pose_index[j]].para_q[0], traj_s[pose_index[j]].para_q[1], traj_s[pose_index[j]].para_q[2]);
                                //                                         Eigen::Quaterniond q_i(traj_s[pose_index[i]].para_q[3], traj_s[pose_index[i]].para_q[0], traj_s[pose_index[i]].para_q[1], traj_s[pose_index[i]].para_q[2]);

                                //                                         Eigen::Vector3d t_i(traj_s[pose_index[i]].para_t[0], traj_s[pose_index[i]].para_t[1], traj_s[pose_index[i]].para_t[2]);
                                //                                         Eigen::Vector3d t_j(traj_s[pose_index[j]].para_t[0], traj_s[pose_index[j]].para_t[1], traj_s[pose_index[j]].para_t[2]);

                                //                                         Eigen::Quaterniond q_j_i = q_j.conjugate() * q_i;
                                //                                         Eigen::Vector3d t_j_i = q_j.conjugate() * (t_i - t_j);

                                //                                         trajectory temp(q_j_i.x(), q_j_i.y(), q_j_i.z(), q_j_i.w(), t_j_i.x(), t_j_i.y(), t_j_i.z());
                                //                                         relative_pose_point[relative_index] = temp;
                                //                                         relative_pose[relative_index] = temp;

                                //                                         problems[relative_index]->AddParameterBlock(relative_pose_point[relative_index].para_q, 4, manifold);
                                //                                         problems[relative_index]->AddParameterBlock(relative_pose_point[relative_index].para_t, 3);

                                //                                         //problems[relative_index].AddParameterBlock(relative_pose_point[relative_index].para_q, 4, manifold);
                                //                                         //problems[relative_index].AddParameterBlock(relative_pose_point[relative_index].para_t, 3);

                                //                                         // problem.AddParameterBlock(relative_pose_point[pose_index[i] + pose_index[j] * pose_size].para_q, 4, manifold);
                                //                                         // problem.AddParameterBlock(relative_pose_point[pose_index[i] + pose_index[j] * pose_size].para_t, 3);
                                //                                     }
                                //                                 }

                                // if(relative_pose_point.find(relative_index)== relative_pose_point.end())
                                // {
                                //     Eigen::Quaterniond q_j(traj_s[pose_index[j]].para_q[3], traj_s[pose_index[j]].para_q[0], traj_s[pose_index[j]].para_q[1], traj_s[pose_index[j]].para_q[2]);
                                //     Eigen::Quaterniond q_i(traj_s[pose_index[i]].para_q[3], traj_s[pose_index[i]].para_q[0], traj_s[pose_index[i]].para_q[1], traj_s[pose_index[i]].para_q[2]);

                                //     Eigen::Vector3d t_i(traj_s[pose_index[i]].para_t[0], traj_s[pose_index[i]].para_t[1], traj_s[pose_index[i]].para_t[2]);
                                //     Eigen::Vector3d t_j(traj_s[pose_index[j]].para_t[0], traj_s[pose_index[j]].para_t[1], traj_s[pose_index[j]].para_t[2]);

                                //     Eigen::Quaterniond q_j_i = q_j.conjugate() * q_i;
                                //     Eigen::Vector3d t_j_i = q_j.conjugate() * (t_i - t_j);

                                //     trajectory temp(q_j_i.x(), q_j_i.y(), q_j_i.z(), q_j_i.w(), t_j_i.x(), t_j_i.y(), t_j_i.z());
                                //     relative_pose_point[relative_index] = temp;
                                // }

                                PointCloudPtr source(new PointCloud);
                                PointCloudPtr target(new PointCloud);
                                for (int k = 0; k < iter->second->pts_orig[pose_index[j]].size(); k++)
                                {
                                    PointType temp(iter->second->pts_orig[pose_index[j]][k].x(), iter->second->pts_orig[pose_index[j]][k].y(), iter->second->pts_orig[pose_index[j]][k].z());
                                    target->points.push_back(temp);
                                }

                                pcl::KdTreeFLANN<PointType> kdtree_target;
                                kdtree_target.setInputCloud(target);

                                for (int k = 0; k < iter->second->pts_orig[pose_index[i]].size(); k++)
                                {
                                    PointType temp(iter->second->pts_orig[pose_index[i]][k].x(), iter->second->pts_orig[pose_index[i]][k].y(), iter->second->pts_orig[pose_index[i]][k].z());
                                    PointType temp_trans;
                                    // tool::TransformPoint(temp, temp_trans, relative_pose_point[relative_index]);
                                    tool::TransformPoint(temp, temp_trans, relative_pose_point2point[relative_index]);
                                    source->points.push_back(temp);
                                }
                                pcl::KdTreeFLANN<PointType> kdtree_source;
                                kdtree_source.setInputCloud(source);

                                // pcl::registration::CorrespondenceEstimation<PointType,PointType> corr_es;
                                // pcl::Correspondences temp;
                                // corr_es.setInputSource(source);
                                // corr_es.setInputTarget(target);
                                // corr_es.determineReciprocalCorrespondences(temp,0.1);

                                // for(int k=0;k<temp.size();k++)
                                // {
                                //     Eigen::Vector3d curr(source->points[temp[k].index_query].x,source->points[temp[k].index_query].y,source->points[temp[k].index_query].z);
                                //     Eigen::Vector3d map(target->points[temp[k].index_match].x,target->points[temp[k].index_match].y,target->points[temp[k].index_match].z);

                                //     //ceres::CostFunction *cost_function = new Point2PointOptimize(curr,map);
                                //     //problem.AddResidualBlock(cost_function,nullptr,relative_pose[pose_index[i] + pose_index[j] * pose_size].para_q,relative_pose[pose_index[i] + pose_index[j] * pose_size].para_t);
                                // }

                                for (int k = 0; k < iter->second->pts_orig[pose_index[i]].size(); k++)
                                {

                                    PointType curr(source->points[k].x, source->points[k].y, source->points[k].z);
                                    // if(!std::isfinite(curr.x)||!std::isfinite(curr.y)||!std::isfinite(curr.z))
                                    // {
                                    //     std::cout<<"point: "<<curr.x<<curr.y<<curr.z<<"\n";
                                    //     continue;
                                    // }
                                    // Eigen::Vector3d map(target->points[temp[k].index_match].x,target->points[temp[k].index_match].y,target->points[temp[k].index_match].z);
                                    std::vector<float> distances_t;
                                    std::vector<int> indices_t;
                                    std::vector<float> distances_s;
                                    std::vector<int> indices_s;
                                    kdtree_target.nearestKSearch(curr, 1, indices_t, distances_t);
                                    kdtree_source.nearestKSearch(target->points[indices_t[0]], 1, indices_s, distances_s);
                                    if (indices_s[0] == k && distances_t[0] < 0.1)
                                    {
                                        Eigen::Vector3d cp(iter->second->pts_orig[pose_index[i]][k].x(), iter->second->pts_orig[pose_index[i]][k].y(), iter->second->pts_orig[pose_index[i]][k].z());
                                        Eigen::Vector3d map(target->points[indices_t[0]].x, target->points[indices_t[0]].y, target->points[indices_t[0]].z);
                                        ceres::CostFunction *cost_function = Point2PointAuto::Create(cp, map);
                                        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

#pragma omp critical
                                        {
                                            //problems[relative_index]->AddResidualBlock(cost_function, loss_function, relative_pose_point[relative_index].para_q, relative_pose_point[relative_index].para_t);
                                            problems[relative_index]->AddResidualBlock(cost_function, loss_function, relative_pose[relative_index].para_q, relative_pose[relative_index].para_t);
                                            // problems[relative_index].AddResidualBlock(cost_function, nullptr, relative_pose_point[relative_index].para_q, relative_pose_point[relative_index].para_t);

                                            // problem.AddResidualBlock(cost_function, nullptr, relative_pose_point[pose_index[i] + pose_index[j] * pose_size].para_q, relative_pose_point[pose_index[i] + pose_index[j] * pose_size].para_t);
                                        }
                                    }
                                    else
                                        continue;

                                    // ceres::CostFunction *cost_function = new Point2PointOptimize(curr,map);
                                    // problem.AddResidualBlock(cost_function,nullptr,relative_pose[pose_index[i] + pose_index[j] * pose_size].para_q,relative_pose[pose_index[i] + pose_index[j] * pose_size].para_t);
                                }
                                //     }
                                // }
                            }
                        }
                    }
                }
            }
        }
    }

    LOG(INFO)<<"finished\n";
    LOG(INFO)<<"number of voxels that overlap > 3 : "<<overlap_count<<"\n";
    LOG(INFO)<<"plane optimization num: "<<plane_count<<"\n";
    LOG(INFO)<<"point optimization num:  "<<notplane_count<<"\n";

    if(last_plane_count>=plane_count)
    {
        solve_flag = false;
        LOG(INFO)<<"termination criteria is satisfied.\n";
        return;
    }

    LOG(INFO)<<"problem size:"<<problems.size()<<"\n";
    LOG(INFO)<<"relative pose size: "<<relative_pose_graph.size()<<"\n";
    LOG(INFO) <<"solving problem...\n";

    // ceres::Solve(options,&problem,&sum);
#pragma omp parallel
{
#pragma omp single
    {
            for (auto iter = problems.begin();iter!=problems.end();iter++)
#pragma omp task
            {
                
                //  if(iter->second->NumResidualBlocks()>200000)
                //  {
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
                    options.max_num_iterations = 8;
                    options.num_threads = 1;
                    ceres::Solver::Summary sum;
                    ceres::Solve(options, iter->second, &sum);
                    
                // }
                // std::vector<const double*> cov_blocks;
                // cov_blocks.push_back(relative_pose[iter->first].para_q);
                // cov_blocks.push_back(relative_pose[iter->first].para_t);
                // //cov_blocks.push_back();

                // //std::vector<std::pair<const double*,const double*>> cov_blocks;
                // //cov_blocks.push_back(std::make_pair(relative_pose[iter->first].para_q,relative_pose[iter->first].para_q));
                // //cov_blocks.push_back(std::make_pair(relative_pose[iter->first].para_t,relative_pose[iter->first].para_t));
                // //cov_blocks.push_back(std::make_pair(relative_pose[iter->first].para_q,relative_pose[iter->first].para_t));
                // ceres::Covariance::Options cov_options;
                // cov_options.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
                
                // ceres::Covariance cov(cov_options);

                // CHECK(cov.Compute(cov_blocks,iter->second));
                //cov.GetCovarianceBlockInTangentSpace(relative_pose[iter->first].para_q,relative_pose[iter->first].para_q,relative_pose[iter->first].covariance_q);
                //cov.GetCovarianceBlock(relative_pose[iter->first].para_t,relative_pose[iter->first].para_t,relative_pose[iter->first].covariance_t);
                //cov.GetCovarianceMatrixInTangentSpace(cov_blocks,relative_pose[iter->first].covariance_q_t);
                delete iter->second;
            }
    }
}
    //problems.clear();

    double time =t_solver.toc();
    LOG(INFO) <<"relative poses are sovled in "<<time<<" secs\n";
    cost_time +=time;
    // if(plane->points.size()!=0)
    // {
    //     pcl::io::savePCDFileBinary("/home/sgg-whu/TLS/EVPA/src/data/result/plane.pcd",*plane);
    // }
    // if(notplane->points.size()!=0)
    // {
    //     pcl::io::savePCDFileBinary("/home/sgg-whu/TLS/EVPA/src/data/result/notplane.pcd",*notplane);
    // }
}

void EVPA::PrintRelativePoseGraph()
{
    std::vector<int> graph(pose_size*pose_size,0);
    for(auto iter:relative_pose_graph)
    {
        graph[iter.second.second*pose_size+iter.second.first] = 1;
        graph[iter.second.first*pose_size+iter.second.second] = 1;
    }

    for(int i =0;i<pose_size;i++)
    {
        for(int j=0;j<pose_size;j++)
        {
            std::cout<<graph[i*pose_size+j]<<"    ";
        }
        std::cout<<"\n";
    }
}

void EVPA::PoseGraphOptimization(std::vector<trajectory> &traj_s)
{
    if(!solve_flag)
        return;
    
    LOG(INFO)<<"start the pose graph optimization...\n";
    TicToc pose_grpah_solver;
    pose_grpah_solver.tic();
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 200;
    options.num_threads = 16;
    ceres::Manifold *manifold = new ceres::EigenQuaternionManifold();
    ceres::Solver::Summary sum;

    
    //point2plane factor
    for(auto iter:relative_pose)
    {
        std::pair<int,int> col_raw = relative_pose_graph[iter.first];
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

        // Eigen::Quaterniond q_esti(iter.second.para_q[3],iter.second.para_q[0],iter.second.para_q[1],iter.second.para_q[2]);
        // Eigen::Vector3d t_esti(iter.second.para_t[0],iter.second.para_t[1],iter.second.para_t[2]);
        // Eigen::Quaterniond q_init(relative_pose_point[iter.first].para_q[3],relative_pose_point[iter.first].para_q[0],relative_pose_point[iter.first].para_q[1],relative_pose_point[iter.first].para_q[2]);
        // Eigen::Vector3d t_init(relative_pose_point[iter.first].para_t[0],relative_pose_point[iter.first].para_t[1],relative_pose_point[iter.first].para_t[2]);
        
        // double distance = (t_esti-t_init).norm();
        // double angle = Eigen::AngleAxisd(q_esti.conjugate()*q_init).angle();
        // if(angle>0.001308 || distance > 0.2)
        //     continue;
        
        //ceres::CostFunction *cost_function = PoseGraphOptimizeAuto::Create(iter.second.para_q,iter.second.para_t,iter.second.covariance_q_t); covariance
        //ceres::CostFunction *cost_function = PoseGraphOptimizeAuto::Create(iter.second.para_q,iter.second.para_t);
        ceres::CostFunction *cost_function = new PoseGraphOptimize(iter.second.para_q,iter.second.para_t);
        problem.AddResidualBlock(cost_function,loss_function,traj_s[col_raw.second].para_q,traj_s[col_raw.second].para_t,traj_s[col_raw.first].para_q,traj_s[col_raw.first].para_t);
        problem.SetManifold(traj_s[col_raw.second].para_q,manifold);
        problem.SetManifold(traj_s[col_raw.first].para_q,manifold);
    }

    // //point2point factor
    // for(auto iter:relative_pose_point2point)
    // {
    //     std::pair<int,int> col_raw = relative_pose_graph[iter.first];
    //     ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

    //     Eigen::Quaterniond q_esti(iter.second.para_q[3],iter.second.para_q[0],iter.second.para_q[1],iter.second.para_q[2]);
    //     Eigen::Vector3d t_esti(iter.second.para_t[0],iter.second.para_t[1],iter.second.para_t[2]);
    //     Eigen::Quaterniond q_init(relative_pose_point[iter.first].para_q[3],relative_pose_point[iter.first].para_q[0],relative_pose_point[iter.first].para_q[1],relative_pose_point[iter.first].para_q[2]);
    //     Eigen::Vector3d t_init(relative_pose_point[iter.first].para_t[0],relative_pose_point[iter.first].para_t[1],relative_pose_point[iter.first].para_t[2]);
        
    //     double distance = (t_esti-t_init).norm();
    //     double angle = Eigen::AngleAxisd(q_esti.conjugate()*q_init).angle();
    //     if(angle>0.001308 || distance > 0.2)
    //         continue;

    //     ceres::CostFunction *cost_function = PoseGraphOptimizeAuto::Create(iter.second.para_q,iter.second.para_t);
    //     problem.AddResidualBlock(cost_function,loss_function,traj_s[col_raw.second].para_q,traj_s[col_raw.second].para_t,traj_s[col_raw.first].para_q,traj_s[col_raw.first].para_t);
    //     problem.SetManifold(traj_s[col_raw.second].para_q,manifold);
    //     problem.SetManifold(traj_s[col_raw.first].para_q,manifold);
    // }

    problem.SetParameterBlockConstant(traj_s[0].para_q);
    problem.SetParameterBlockConstant(traj_s[0].para_t);

    ceres::Solve(options,&problem,&sum);
    
    double time = pose_grpah_solver.toc();
    LOG(INFO)<<"pose graph optimization is done in "<<pose_grpah_solver.toc()<<" secs.\n\n";

    cost_time+=time;
}

void EVPA::init()
{
    relative_pose.clear();
    relative_pose_graph.clear();

    for(auto voxel:Voxelized_map)
    {
        delete voxel.second;
    }
    Voxelized_map.clear();
    Voxelized_map_plane_visualize.clear();
    pose_index_each_voxel.clear();
}



void EVPA::SolveWithICP(std::vector<PointCloudPtr> &pcs,std::vector<trajectory> &traj_s)
{

    std::cout<<"begin to solve with icp...\n";
    for (std::unordered_map<VoxelLocation, Voxel *>::iterator iter = Voxelized_map.begin(); iter != Voxelized_map.end(); iter++)
    {

        // 寻找重叠部分的体素
        std::vector<int> pose_index;
        // std::unordered_map<int,std::pair<int,int>> relative_pose_graph_this_voxel;

        for (int i = 0; i < pose_size; i++)
        {
            if (iter->second->pts_trans[i].size() < 30)
                    continue;

            pose_index.push_back(i);
        }
        // 如果有重叠部分的体素，则构建因子图,从后往前（root）配准

        // if (pose_index.size() < 2)
        //     continue;

        //

        // plane factors
        if (pose_index.size() >= 2)
        {

            for (int i = pose_index.size() - 1; i >= 1; i--)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    if(pose_index[i]-pose_index[j]>=reliable_region)
                                break;

                    int relative_index = pose_index[i] + pose_index[j] * pose_size;
                    if (relative_pose_graph.find(relative_index) == relative_pose_graph.end())
                    {
                        relative_pose_graph[relative_index] = std::make_pair(pose_index[i], pose_index[j]);

                        Eigen::Quaterniond q_j(traj_s[pose_index[j]].para_q[3], traj_s[pose_index[j]].para_q[0], traj_s[pose_index[j]].para_q[1], traj_s[pose_index[j]].para_q[2]);
                        Eigen::Quaterniond q_i(traj_s[pose_index[i]].para_q[3], traj_s[pose_index[i]].para_q[0], traj_s[pose_index[i]].para_q[1], traj_s[pose_index[i]].para_q[2]);

                        Eigen::Vector3d t_i(traj_s[pose_index[i]].para_t[0], traj_s[pose_index[i]].para_t[1], traj_s[pose_index[i]].para_t[2]);
                        Eigen::Vector3d t_j(traj_s[pose_index[j]].para_t[0], traj_s[pose_index[j]].para_t[1], traj_s[pose_index[j]].para_t[2]);

                        Eigen::Quaterniond q_j_i = q_j.conjugate() * q_i;
                        Eigen::Vector3d t_j_i = q_j.conjugate() * (t_i - t_j);

                        trajectory temp(q_j_i.x(), q_j_i.y(), q_j_i.z(), q_j_i.w(), t_j_i.x(), t_j_i.y(), t_j_i.z());

                        relative_pose[relative_index] = temp;
                    }
                }
            }
        }
    }

    std::cout<<"relative pose size: "<<relative_pose.size()<<"\n";
    TicToc t_solver;
    t_solver.tic();
#pragma omp parallel
    {
#pragma omp single
        {
            for (auto iter : relative_pose_graph)
#pragma omp task
            {
                pcl::IterativeClosestPoint<PointType, PointType> icp;
                pcl::PointCloud<PointType> res;
                Eigen::Quaterniond q_r(relative_pose[iter.first].para_q[3], relative_pose[iter.first].para_q[0], relative_pose[iter.first].para_q[1], relative_pose[iter.first].para_q[2]);
                Eigen::Affine3d aff;
                aff.linear() = q_r.toRotationMatrix();
                aff.translation() << relative_pose[iter.first].para_t[0], relative_pose[iter.first].para_t[1], relative_pose[iter.first].para_t[2];
                icp.setInputSource(pcs[iter.second.first]);
                icp.setInputTarget(pcs[iter.second.second]);
                icp.setMaxCorrespondenceDistance(0.1);
                icp.setUseReciprocalCorrespondences(true);
                icp.setMaximumIterations(1);
                icp.align(res, aff.matrix().cast<float>());

                Eigen::Affine3d aff_res;
                aff_res.matrix() = icp.getFinalTransformation().cast<double>();
                Eigen::Quaterniond q(aff_res.rotation());
                relative_pose[iter.first].para_q[0] = q.x();
                relative_pose[iter.first].para_q[1] = q.y();
                relative_pose[iter.first].para_q[2] = q.z();
                relative_pose[iter.first].para_q[3] = q.w();

                relative_pose[iter.first].para_t[0] = aff_res.translation()(0, 0);
                relative_pose[iter.first].para_t[1] = aff_res.translation()(1, 0);
                relative_pose[iter.first].para_t[2] = aff_res.translation()(2, 0);
            }
        }
    }
    double time = t_solver.toc();
    std::cout<<"finished\n";
    std::cout <<"relative poses are sovled in "<<time<<" secs\n";
    cost_time +=time;

}

void EVPA::SolveWithICPNormal(std::vector<PointCloudPtr> &pcs,std::vector<trajectory> &traj_s)
{
    std::cout<<"begin to solve with point to plane icp...\n";
    for (std::unordered_map<VoxelLocation, Voxel *>::iterator iter = Voxelized_map.begin(); iter != Voxelized_map.end(); iter++)
    {

        // 寻找重叠部分的体素
        std::vector<int> pose_index;
        // std::unordered_map<int,std::pair<int,int>> relative_pose_graph_this_voxel;

        for (int i = 0; i < pose_size; i++)
        {
            if (iter->second->pts_trans[i].size() < 30)
                    continue;

            pose_index.push_back(i);
        }
        // 如果有重叠部分的体素，则构建因子图,从后往前（root）配准

        // if (pose_index.size() < 2)
        //     continue;

        //

        // plane factors
        if (pose_index.size() >= 2)
        {

            for (int i = pose_index.size() - 1; i >= 1; i--)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    if(pose_index[i]-pose_index[j]>=reliable_region)
                        break;
                    int relative_index = pose_index[i] + pose_index[j] * pose_size;
                    if (relative_pose_graph.find(relative_index) == relative_pose_graph.end())
                    {
                        relative_pose_graph[relative_index] = std::make_pair(pose_index[i], pose_index[j]);

                        Eigen::Quaterniond q_j(traj_s[pose_index[j]].para_q[3], traj_s[pose_index[j]].para_q[0], traj_s[pose_index[j]].para_q[1], traj_s[pose_index[j]].para_q[2]);
                        Eigen::Quaterniond q_i(traj_s[pose_index[i]].para_q[3], traj_s[pose_index[i]].para_q[0], traj_s[pose_index[i]].para_q[1], traj_s[pose_index[i]].para_q[2]);

                        Eigen::Vector3d t_i(traj_s[pose_index[i]].para_t[0], traj_s[pose_index[i]].para_t[1], traj_s[pose_index[i]].para_t[2]);
                        Eigen::Vector3d t_j(traj_s[pose_index[j]].para_t[0], traj_s[pose_index[j]].para_t[1], traj_s[pose_index[j]].para_t[2]);

                        Eigen::Quaterniond q_j_i = q_j.conjugate() * q_i;
                        Eigen::Vector3d t_j_i = q_j.conjugate() * (t_i - t_j);

                        trajectory temp(q_j_i.x(), q_j_i.y(), q_j_i.z(), q_j_i.w(), t_j_i.x(), t_j_i.y(), t_j_i.z());

                        relative_pose[relative_index] = temp;
                    }
                }
            }
        }
    }
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> pcs_nomals(pcs.size());
    std::cout<<"relative pose size: "<<relative_pose.size()<<"\n";
    TicToc t_solver;
    t_solver.tic();

#pragma omp parallel
    {
#pragma omp single
        {
            for (int i=0;i<pcs.size();i++)
#pragma omp task
            {
                pcl::PointCloud<pcl::Normal>::Ptr pcs_nomal(new pcl::PointCloud<pcl::Normal>());
                pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne;
                //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
                ne.setInputCloud(pcs[i]);
                ne.setRadiusSearch(0.1);
                //ne.setSearchMethod(tree);
                ne.compute(*pcs_nomal);
                pcl::PointCloud<pcl::PointNormal>::Ptr out(new pcl::PointCloud<pcl::PointNormal>());

                pcl::concatenateFields(*pcs[i],*pcs_nomal,*out);
                pcs_nomals[i]=out;
            }
        }
    }

#pragma omp parallel
    {
#pragma omp single
        {
            for (auto iter : relative_pose_graph)
#pragma omp task
            {
                pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal,pcl::PointNormal>::Ptr trans(new pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal,pcl::PointNormal>());
                pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
                pcl::PointCloud<pcl::PointNormal> res;
                Eigen::Quaterniond q_r(relative_pose[iter.first].para_q[3], relative_pose[iter.first].para_q[0], relative_pose[iter.first].para_q[1], relative_pose[iter.first].para_q[2]);
                Eigen::Affine3d aff;
                aff.linear() = q_r.toRotationMatrix();
                aff.translation() << relative_pose[iter.first].para_t[0], relative_pose[iter.first].para_t[1], relative_pose[iter.first].para_t[2];
                icp.setInputSource(pcs_nomals[iter.second.first]);
                icp.setInputTarget(pcs_nomals[iter.second.second]);
                icp.setMaxCorrespondenceDistance(0.1);
                icp.setUseReciprocalCorrespondences(true);
                icp.setTransformationEstimation(trans);
                icp.setMaximumIterations(1);
                icp.align(res, aff.matrix().cast<float>());

                Eigen::Affine3d aff_res;
                aff_res.matrix() = icp.getFinalTransformation().cast<double>();
                Eigen::Quaterniond q(aff_res.rotation());
                relative_pose[iter.first].para_q[0] = q.x();
                relative_pose[iter.first].para_q[1] = q.y();
                relative_pose[iter.first].para_q[2] = q.z();
                relative_pose[iter.first].para_q[3] = q.w();

                relative_pose[iter.first].para_t[0] = aff_res.translation()(0, 0);
                relative_pose[iter.first].para_t[1] = aff_res.translation()(1, 0);
                relative_pose[iter.first].para_t[2] = aff_res.translation()(2, 0);
            }
        }
    }

    double time = t_solver.toc();
    std::cout<<"finished\n";
    std::cout <<"relative poses are sovled in "<<time<<" secs\n";
    cost_time +=time;
}

void EVPA::SolveWithLum(std::vector<PointCloudPtr> &pcs,std::vector<trajectory> &traj_s)
{

    std::cout<<"begin to solve with lum...\n";
    for (std::unordered_map<VoxelLocation, Voxel *>::iterator iter = Voxelized_map.begin(); iter != Voxelized_map.end(); iter++)
    {

        // 寻找重叠部分的体素
        std::vector<int> pose_index;
        // std::unordered_map<int,std::pair<int,int>> relative_pose_graph_this_voxel;

        for (int i = 0; i < pose_size; i++)
        {
            if (iter->second->pts_trans[i].size() < 30)
                    continue;

            pose_index.push_back(i);
        }
        // 如果有重叠部分的体素，则构建因子图,从后往前（root）配准

        // if (pose_index.size() < 2)
        //     continue;

        //

        // plane factors
        if (pose_index.size() >= 2)
        {

            for (int i = pose_index.size() - 1; i >= 1; i--)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    int relative_index = pose_index[i] + pose_index[j] * pose_size;
                    if (relative_pose_graph.find(relative_index) == relative_pose_graph.end())
                    {
                        if(pose_index[i]-pose_index[j]>=reliable_region)
                            break;
                        relative_pose_graph[relative_index] = std::make_pair(pose_index[i], pose_index[j]);

                        Eigen::Quaterniond q_j(traj_s[pose_index[j]].para_q[3], traj_s[pose_index[j]].para_q[0], traj_s[pose_index[j]].para_q[1], traj_s[pose_index[j]].para_q[2]);
                        Eigen::Quaterniond q_i(traj_s[pose_index[i]].para_q[3], traj_s[pose_index[i]].para_q[0], traj_s[pose_index[i]].para_q[1], traj_s[pose_index[i]].para_q[2]);

                        Eigen::Vector3d t_i(traj_s[pose_index[i]].para_t[0], traj_s[pose_index[i]].para_t[1], traj_s[pose_index[i]].para_t[2]);
                        Eigen::Vector3d t_j(traj_s[pose_index[j]].para_t[0], traj_s[pose_index[j]].para_t[1], traj_s[pose_index[j]].para_t[2]);

                        Eigen::Quaterniond q_j_i = q_j.conjugate() * q_i;
                        Eigen::Vector3d t_j_i = q_j.conjugate() * (t_i - t_j);

                        trajectory temp(q_j_i.x(), q_j_i.y(), q_j_i.z(), q_j_i.w(), t_j_i.x(), t_j_i.y(), t_j_i.z());

                        relative_pose[relative_index] = temp;
                    }
                }
            }
        }
    }
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> pcs_nomals;
    std::cout<<"relative pose size: "<<relative_pose.size()<<"\n";
    TicToc t_solver;
    t_solver.tic();
    pcl::registration::LUM<PointType> lum;

    for(int i =0;i<pcs.size();i++)
    {
        Eigen::Quaternionf q_(traj_s[i].para_q[3],traj_s[i].para_q[0],traj_s[i].para_q[1],traj_s[i].para_q[2]);
        Eigen::Affine3f traj;
        traj.linear() = q_.toRotationMatrix();
        traj.translation() << traj_s[i].para_t[0],traj_s[i].para_t[1],traj_s[i].para_t[2];
        Eigen::Vector6f pose;
        pcl::getTranslationAndEulerAngles(traj,pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]);

        lum.addPointCloud(pcs[i],pose);
    }

    lum.setMaxIterations(1);
    //lum.setConvergenceThreshold(0.01);

    //std::vector<pcl::Correspondences> corrs;
#pragma omp parallel
    {
#pragma omp single
        {
            for (auto iter : relative_pose_graph)
#pragma omp task
            {
                pcl::registration::CorrespondenceEstimation<PointType, PointType> ce;
                pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>());
                Eigen::Quaternionf trans(relative_pose[iter.first].para_q[3],relative_pose[iter.first].para_q[0],relative_pose[iter.first].para_q[1],relative_pose[iter.first].para_q[2]);
                Eigen::Affine3f aff;
                aff.linear() = trans.toRotationMatrix();
                aff.translation() << relative_pose[iter.first].para_t[0],relative_pose[iter.first].para_t[1],relative_pose[iter.first].para_t[2];
                pcl::transformPointCloud(*pcs[iter.second.first],*temp,aff.matrix());
                pcl::CorrespondencesPtr corr(new pcl::Correspondences);
                ce.setInputSource(temp);
                ce.setInputTarget(pcs[iter.second.second]);
                ce.determineReciprocalCorrespondences(*corr, 0.1);
                lum.setCorrespondences(iter.second.first, iter.second.second, corr);
            }
        }
    }

    std::cout<<"calculate correspondences done..\n";
    lum.compute();

    for(int i = 0;i<pcs.size();i++)
    {
        Eigen::Quaternionf q(lum.getTransformation(i).rotation());

        traj_s[i].para_q[0] = q.x();
        traj_s[i].para_q[1] = q.y();
        traj_s[i].para_q[2] = q.z();
        traj_s[i].para_q[3] = q.w();

        traj_s[i].para_t[0] = lum.getTransformation(i).translation().x();
        traj_s[i].para_t[1] = lum.getTransformation(i).translation().y();
        traj_s[i].para_t[2] = lum.getTransformation(i).translation().z();

    }
    double time = t_solver.toc();
    std::cout<<"finished\n";
    std::cout <<"relative poses are sovled in "<<time<<" secs\n";
    cost_time +=time;
}


void EVPA::SolveWithGICP(std::vector<PointCloudPtr> &pcs,std::vector<trajectory> &traj_s)
{
     std::cout<<"begin to solve with gicp...\n";
    for (std::unordered_map<VoxelLocation, Voxel *>::iterator iter = Voxelized_map.begin(); iter != Voxelized_map.end(); iter++)
    {

        // 寻找重叠部分的体素
        std::vector<int> pose_index;
        // std::unordered_map<int,std::pair<int,int>> relative_pose_graph_this_voxel;

        for (int i = 0; i < pose_size; i++)
        {
            if (iter->second->pts_trans[i].size() < 30)
                    continue;

            pose_index.push_back(i);
        }
        // 如果有重叠部分的体素，则构建因子图,从后往前（root）配准

        // if (pose_index.size() < 2)
        //     continue;

        //

        // plane factors
        if (pose_index.size() >= 2)
        {

            for (int i = pose_index.size() - 1; i >= 1; i--)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    if(pose_index[i]-pose_index[j]>=reliable_region)
                        break;
                    int relative_index = pose_index[i] + pose_index[j] * pose_size;
                    if (relative_pose_graph.find(relative_index) == relative_pose_graph.end())
                    {
                        relative_pose_graph[relative_index] = std::make_pair(pose_index[i], pose_index[j]);

                        Eigen::Quaterniond q_j(traj_s[pose_index[j]].para_q[3], traj_s[pose_index[j]].para_q[0], traj_s[pose_index[j]].para_q[1], traj_s[pose_index[j]].para_q[2]);
                        Eigen::Quaterniond q_i(traj_s[pose_index[i]].para_q[3], traj_s[pose_index[i]].para_q[0], traj_s[pose_index[i]].para_q[1], traj_s[pose_index[i]].para_q[2]);

                        Eigen::Vector3d t_i(traj_s[pose_index[i]].para_t[0], traj_s[pose_index[i]].para_t[1], traj_s[pose_index[i]].para_t[2]);
                        Eigen::Vector3d t_j(traj_s[pose_index[j]].para_t[0], traj_s[pose_index[j]].para_t[1], traj_s[pose_index[j]].para_t[2]);

                        Eigen::Quaterniond q_j_i = q_j.conjugate() * q_i;
                        Eigen::Vector3d t_j_i = q_j.conjugate() * (t_i - t_j);

                        trajectory temp(q_j_i.x(), q_j_i.y(), q_j_i.z(), q_j_i.w(), t_j_i.x(), t_j_i.y(), t_j_i.z());

                        relative_pose[relative_index] = temp;
                    }
                }
            }
        }
    }

    std::cout<<"relative pose size: "<<relative_pose.size()<<"\n";
    TicToc t_solver;
    t_solver.tic();
#pragma omp parallel
    {
#pragma omp single
        {
            for (auto iter : relative_pose_graph)
#pragma omp task
            {
                pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
                pcl::PointCloud<PointType> res;
                Eigen::Quaterniond q_r(relative_pose[iter.first].para_q[3], relative_pose[iter.first].para_q[0], relative_pose[iter.first].para_q[1], relative_pose[iter.first].para_q[2]);
                Eigen::Affine3d aff;
                aff.linear() = q_r.toRotationMatrix();
                aff.translation() << relative_pose[iter.first].para_t[0], relative_pose[iter.first].para_t[1], relative_pose[iter.first].para_t[2];
                icp.setInputSource(pcs[iter.second.first]);
                icp.setInputTarget(pcs[iter.second.second]);
                icp.setMaxCorrespondenceDistance(0.1);
                icp.setUseReciprocalCorrespondences(true);
                icp.setMaximumIterations(1);
                icp.align(res, aff.matrix().cast<float>());

                Eigen::Affine3d aff_res;
                aff_res.matrix() = icp.getFinalTransformation().cast<double>();
                Eigen::Quaterniond q(aff_res.rotation());
                relative_pose[iter.first].para_q[0] = q.x();
                relative_pose[iter.first].para_q[1] = q.y();
                relative_pose[iter.first].para_q[2] = q.z();
                relative_pose[iter.first].para_q[3] = q.w();

                relative_pose[iter.first].para_t[0] = aff_res.translation()(0, 0);
                relative_pose[iter.first].para_t[1] = aff_res.translation()(1, 0);
                relative_pose[iter.first].para_t[2] = aff_res.translation()(2, 0);
            }
        }
    }

    double time = t_solver.toc();
    std::cout<<"finished\n";
    std::cout <<"relative poses are sovled in "<<time<<" secs\n";
    cost_time +=time;

}


void EVPA::VisualizePlaneMatch(std::vector<PointCloudPtr> &pc,std::vector<trajectory> &traj_s)
{
    //transform point cloud
    //pcl::transformPointCloud(pcl)

    pcl::visualization::PCLVisualizer vis;
    for(int i =0;i<pc.size();i++)
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb(pc[i], 178, 222, 207);
        vis.addPointCloud(pc[i],rgb,std::to_string(i));
        std::cout<<"cloud"<<i<<"\n";
    }
    vis.setBackgroundColor(255,255,255);

    //add pose
    std::vector<trajectory> traj;
    // trajectory t0(0,0,0,1,-3.882476199,-11.42862143,0.00373653893);
    // trajectory t1(0,0,0,1,-2.885726015,-7.049986559,-0.6292993469);
    // trajectory t2(0,0,0,1,-5.344721932,-0.7207951372,0.009226195207);
    // traj.push_back(t0);
    // traj.push_back(t1);
    // traj.push_back(t2);
    // for(int i =0;i<traj.size();i++)
    // {
    //     PointType position(traj[i].para_t[0],traj[i].para_t[1],traj[i].para_t[2]);
    //     vis.addSphere(position,0.05,"pose"+std::to_string(i));
    //     std::cout<<"pose"<<i<<"\n";
    // }
    //plane patches
    for(int i=0;i<Voxelized_map_plane_visualize.size();i++)
    {
        PointCloudPtr pc(new PointCloud);
        for(int j=0;j<Voxelized_map_plane_visualize[i]->pts_trans.size();j++)
        {
            if(Voxelized_map_plane_visualize[i]->pts_trans[j].size()==0)
                continue;
            
            for(int k=0;k<Voxelized_map_plane_visualize[i]->pts_trans[j].size();k++)
            {
                PointType pts(Voxelized_map_plane_visualize[i]->pts_trans[j][k].x(),Voxelized_map_plane_visualize[i]->pts_trans[j][k].y(),Voxelized_map_plane_visualize[i]->pts_trans[j][k].z());
                pc->points.push_back(pts);
            }
            
        }
        pcl::visualization::PointCloudColorHandlerRandom<PointType> rgb(pc);
        vis.addPointCloud(pc,rgb,"patches"+std::to_string(i));
        vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,"patches"+std::to_string(i));
        // for(int j=0;j<pose_index_each_voxel[i].size();j++)
        // {
        //     PointType pose(traj[pose_index_each_voxel[i][j]].para_t[0],traj[pose_index_each_voxel[i][j]].para_t[1],traj[pose_index_each_voxel[i][j]].para_t[2]);
        //     PointType patches(Voxelized_map_plane_visualize[i]->voxel_center[0],Voxelized_map_plane_visualize[i]->voxel_center[1],Voxelized_map_plane_visualize[i]->voxel_center[2]);
        //     vis.addLine(pose,patches,"line"+std::to_string(i)+std::to_string(j));
        //     vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,5,"line"+std::to_string(i)+std::to_string(j));
        // }

    }

    //add pose 

    

    

    while(!vis.wasStopped())
    {
        vis.spin();
    }


}


void EVPA::VisualizePointCloud(std::vector<PointCloudPtr> &pc,std::vector<trajectory> &traj_s)
{
    pcl::visualization::PCLVisualizer vis;
    vis.setBackgroundColor(255,255,255);
    for(int i =0;i<pc.size();i++)
    {
        PointCloudPtr pc_trans(new PointCloud);
        Eigen::Quaterniond rotation(traj_s[i].para_q[3],traj_s[i].para_q[0],traj_s[i].para_q[1],traj_s[i].para_q[2]);
        Eigen::Affine3d trans;
        trans.linear() = rotation.matrix();
        trans.translation()<<traj_s[i].para_t[0],traj_s[i].para_t[1],traj_s[i].para_t[2];
        pcl::transformPointCloud(*pc[i],*pc_trans,trans.matrix().cast<float>());


        pcl::visualization::PointCloudColorHandlerRandom<PointType> rgb(pc_trans);
        vis.addPointCloud(pc_trans,rgb,"cloud"+std::to_string(i));
    }

    while(!vis.wasStopped())
    {
        vis.spinOnce();
    }
}