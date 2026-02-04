#pragma once
#include <limits>
#include <vector>
#include <iostream>
#include <time.h>
#include <float.h>
#include "./utils/msg.h"

struct RegionEmbedding
{
	cv::Mat m_hist;
	cv::Mat m_embedding;
	double m_embDot = 0.;
};

class AssignmentProblemSolver
{
public:
	enum TMethod
	{
		optimal,
		many_forbidden_assignments,
		without_forbidden_assignments
	};

	AssignmentProblemSolver() = default;
	~AssignmentProblemSolver() = default;
	void solve(const std::vector<double> &distMatrixIn, size_t nOfRows, size_t nOfColumns, std::vector<int> &assignment, TMethod Method = optimal);
	void calcEmbeddins(std::vector<RegionEmbedding> &regionEmbeddings, const std::vector<sensor_msgs::obj_box_msg> &regions, cv::Mat currFrame);
	void createDistaceMatrix(const std::vector<sensor_msgs::obj_box_msg> &regions_last, const std::vector<sensor_msgs::obj_box_msg> &regions, const std::vector<RegionEmbedding> &regionEmbeddings, std::vector<double> &costMatrix, double maxPossibleCost, double &maxCost);
	// void createDistaceMatrixByIoU(const std::vector<sensor_msgs::obj_box_msg> &regions_last, const std::vector<sensor_msgs::obj_box_msg> &regions, const std::vector<double> &conf, std::vector<double> &costMatrix);
	void createDistaceMatrixByIoU(const std::vector<sensor_msgs::obj_box_msg> &regions_last, const std::vector<sensor_msgs::obj_box_msg> &regions, std::vector<double> &costMatrix);
	void createDistaceMatrixByDistance(const std::vector<sensor_msgs::obj_box_msg> &regions_last, const std::vector<sensor_msgs::obj_box_msg> &regions, std::vector<double> &costMatrix);
	void createDistaceMatrixByGIoU(const std::vector<sensor_msgs::obj_box_msg> &regions_last, const std::vector<sensor_msgs::obj_box_msg> &regions, std::vector<double> &costMatrix);
	cv::RotatedRect calcPredictionEllipse(cv::Size_<double> minRadius) const;
	double computeMinEnclosingVolume(const std::vector<Eigen::Vector3d>& corners1, const std::vector<Eigen::Vector3d>& corners2);

public:
	// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
	void assignmentoptimal(std::vector<int> &assignment, double &cost, const std::vector<double> &distMatrixIn, size_t nOfRows, size_t nOfColumns);
	void buildassignmentvector(std::vector<int> &assignment, bool *starMatrix, size_t nOfRows, size_t nOfColumns);
	void computeassignmentcost(const std::vector<int> &assignment, double &cost, const std::vector<double> &distMatrixIn, size_t nOfRows);
	void step2a(std::vector<int> &assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step2b(std::vector<int> &assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step3_5(std::vector<int> &assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step4(std::vector<int> &assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim, size_t row, size_t col);

	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	void assignmentsuboptimal1(std::vector<int> &assignment, double &cost, const std::vector<double> &distMatrixIn, size_t nOfRows, size_t nOfColumns);
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	void assignmentsuboptimal2(std::vector<int> &assignment, double &cost, const std::vector<double> &distMatrixIn, size_t nOfRows, size_t nOfColumns);
	double getPolygonArea(std::vector<cv::Point2f> const &points);
	std::vector<double> m_distMatrix;
};

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::solve(const std::vector<double> &distMatrixIn, size_t nOfRows, size_t nOfColumns, std::vector<int> &assignment, TMethod Method)
{
	assignment.resize(nOfRows, -1);

	double cost = 0;

	switch (Method)
	{
	case optimal:
		assignmentoptimal(assignment, cost, distMatrixIn, nOfRows, nOfColumns);
		break;

	case many_forbidden_assignments:
		assignmentsuboptimal1(assignment, cost, distMatrixIn, nOfRows, nOfColumns);
		break;

	case without_forbidden_assignments:
		assignmentsuboptimal2(assignment, cost, distMatrixIn, nOfRows, nOfColumns);
		break;
	}

	// return cost;
}
// --------------------------------------------------------------------------
// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentoptimal(std::vector<int> &assignment, double &cost, const std::vector<double> &distMatrixIn, size_t nOfRows, size_t nOfColumns)
{
	// Generate distance cv::Matrix and check cv::Matrix elements positiveness

	// Total elements number
	const size_t nOfElements = nOfRows * nOfColumns;
	// Memory allocation
	m_distMatrix.assign(std::begin(distMatrixIn), std::end(distMatrixIn));
	const double *distMatrixEnd = m_distMatrix.data() + nOfElements;

	// Memory allocation
	bool *coveredColumns = (bool *)calloc(nOfColumns, sizeof(bool));
	bool *coveredRows = (bool *)calloc(nOfRows, sizeof(bool));
	bool *starMatrix = (bool *)calloc(nOfElements, sizeof(bool));
	bool *primeMatrix = (bool *)calloc(nOfElements, sizeof(bool));
	bool *newStarMatrix = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

	/* preliminary steps */
	if (nOfRows <= nOfColumns)
	{
		for (size_t row = 0; row < nOfRows; ++row)
		{
			/* find the smallest element in the row */
			double *distMatrixTemp = m_distMatrix.data() + row;
			double minValue = *distMatrixTemp;
			distMatrixTemp += nOfRows;
			while (distMatrixTemp < distMatrixEnd)
			{
				double value = *distMatrixTemp;
				if (value < minValue)
					minValue = value;

				distMatrixTemp += nOfRows;
			}
			/* subtract the smallest element from each element of the row */
			distMatrixTemp = m_distMatrix.data() + row;
			while (distMatrixTemp < distMatrixEnd)
			{
				*distMatrixTemp -= minValue;
				distMatrixTemp += nOfRows;
			}
		}
		/* Steps 1 and 2a */
		for (size_t row = 0; row < nOfRows; ++row)
		{
			for (size_t col = 0; col < nOfColumns; ++col)
			{
				if (m_distMatrix[row + nOfRows * col] == 0)
				{
					if (!coveredColumns[col])
					{
						starMatrix[row + nOfRows * col] = true;
						coveredColumns[col] = true;
						break;
					}
				}
			}
		}
	}
	else /* if(nOfRows > nOfColumns) */
	{
		for (size_t col = 0; col < nOfColumns; ++col)
		{
			/* find the smallest element in the column */
			double *distMatrixTemp = m_distMatrix.data() + nOfRows * col;
			double *columnEnd = distMatrixTemp + nOfRows;
			double minValue = *distMatrixTemp++;
			while (distMatrixTemp < columnEnd)
			{
				double value = *distMatrixTemp++;
				if (value < minValue)
					minValue = value;
			}
			/* subtract the smallest element from each element of the column */
			distMatrixTemp = m_distMatrix.data() + nOfRows * col;
			while (distMatrixTemp < columnEnd)
			{
				*distMatrixTemp++ -= minValue;
			}
		}
		/* Steps 1 and 2a */
		for (size_t col = 0; col < nOfColumns; ++col)
		{
			for (size_t row = 0; row < nOfRows; ++row)
			{
				if (m_distMatrix[row + nOfRows * col] == 0)
				{
					if (!coveredRows[row])
					{
						starMatrix[row + nOfRows * col] = true;
						coveredColumns[col] = true;
						coveredRows[row] = true;
						break;
					}
				}
			}
		}

		for (size_t row = 0; row < nOfRows; ++row)
		{
			coveredRows[row] = false;
		}
	}
	/* move to step 2b */
	step2b(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, (nOfRows <= nOfColumns) ? nOfRows : nOfColumns);
	/* compute cost and remove invalid assignments */
	computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
	/* free allocated memory */
	free(coveredColumns);
	free(coveredRows);
	free(starMatrix);
	free(primeMatrix);
	free(newStarMatrix);
	return;
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::buildassignmentvector(std::vector<int> &assignment, bool *starMatrix, size_t nOfRows, size_t nOfColumns)
{
	for (size_t row = 0; row < nOfRows; ++row)
	{
		for (size_t col = 0; col < nOfColumns; ++col)
		{
			if (starMatrix[row + nOfRows * col])
			{
				assignment[row] = static_cast<int>(col);
				break;
			}
		}
	}
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::computeassignmentcost(const std::vector<int> &assignment, double &cost, const std::vector<double> &distMatrixIn, size_t nOfRows)
{
	for (size_t row = 0; row < nOfRows; ++row)
	{
		const int col = assignment[row];
		if (col >= 0)
			cost += distMatrixIn[row + nOfRows * col];
	}
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2a(std::vector<int> &assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim)
{
	bool *starMatrixTemp, *columnEnd;
	/* cover every column containing a starred zero */
	for (size_t col = 0; col < nOfColumns; ++col)
	{
		starMatrixTemp = starMatrix + nOfRows * col;
		columnEnd = starMatrixTemp + nOfRows;
		while (starMatrixTemp < columnEnd)
		{
			if (*starMatrixTemp++)
			{
				coveredColumns[col] = true;
				break;
			}
		}
	}
	/* move to step 3 */
	step2b(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2b(std::vector<int> &assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim)
{
	/* count covered columns */
	size_t nOfCoveredColumns = 0;
	for (size_t col = 0; col < nOfColumns; ++col)
	{
		if (coveredColumns[col])
			nOfCoveredColumns++;
	}
	if (nOfCoveredColumns == minDim) // algorithm finished
		buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
	else // move to step 3
		step3_5(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step3_5(std::vector<int> &assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim)
{
	for (;;)
	{
		/* step 3 */
		bool zerosFound = true;
		while (zerosFound)
		{
			zerosFound = false;
			for (size_t col = 0; col < nOfColumns; ++col)
			{
				if (!coveredColumns[col])
				{
					for (size_t row = 0; row < nOfRows; ++row)
					{
						if ((!coveredRows[row]) && (m_distMatrix[row + nOfRows * col] == 0))
						{
							/* prime zero */
							primeMatrix[row + nOfRows * col] = true;
							/* find starred zero in current row */
							size_t starCol = 0;
							for (; starCol < nOfColumns; ++starCol)
							{
								if (starMatrix[row + nOfRows * starCol])
									break;
							}
							if (starCol == nOfColumns) /* no starred zero found */
							{
								/* move to step 4 */
								step4(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
								return;
							}
							else
							{
								coveredRows[row] = true;
								coveredColumns[starCol] = false;
								zerosFound = true;
								break;
							}
						}
					}
				}
			}
		}
		/* step 5 */
		double h = DBL_MAX;
		for (size_t row = 0; row < nOfRows; ++row)
		{
			if (!coveredRows[row])
			{
				for (size_t col = 0; col < nOfColumns; ++col)
				{
					if (!coveredColumns[col])
					{
						const double value = m_distMatrix[row + nOfRows * col];
						if (value < h)
							h = value;
					}
				}
			}
		}
		/* add h to each covered row */
		for (size_t row = 0; row < nOfRows; ++row)
		{
			if (coveredRows[row])
			{
				for (size_t col = 0; col < nOfColumns; ++col)
				{
					m_distMatrix[row + nOfRows * col] += h;
				}
			}
		}
		/* subtract h from each uncovered column */
		for (size_t col = 0; col < nOfColumns; ++col)
		{
			if (!coveredColumns[col])
			{
				for (size_t row = 0; row < nOfRows; ++row)
				{
					m_distMatrix[row + nOfRows * col] -= h;
				}
			}
		}
	}
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step4(std::vector<int> &assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim, size_t row, size_t col)
{
	const size_t nOfElements = nOfRows * nOfColumns;
	/* generate temporary copy of starMatrix */
	for (size_t n = 0; n < nOfElements; ++n)
	{
		newStarMatrix[n] = starMatrix[n];
	}
	/* star current zero */
	newStarMatrix[row + nOfRows * col] = true;
	/* find starred zero in current column */
	size_t starCol = col;
	size_t starRow = 0;
	for (; starRow < nOfRows; ++starRow)
	{
		if (starMatrix[starRow + nOfRows * starCol])
			break;
	}
	while (starRow < nOfRows)
	{
		/* unstar the starred zero */
		newStarMatrix[starRow + nOfRows * starCol] = false;
		/* find primed zero in current row */
		size_t primeRow = starRow;
		size_t primeCol = 0;
		for (; primeCol < nOfColumns; ++primeCol)
		{
			if (primeMatrix[primeRow + nOfRows * primeCol])
				break;
		}
		/* star the primed zero */
		newStarMatrix[primeRow + nOfRows * primeCol] = true;
		/* find starred zero in current column */
		starCol = primeCol;
		for (starRow = 0; starRow < nOfRows; ++starRow)
		{
			if (starMatrix[starRow + nOfRows * starCol])
				break;
		}
	}
	/* use temporary copy as new starMatrix */
	/* delete all primes, uncover all rows */
	for (size_t n = 0; n < nOfElements; ++n)
	{
		primeMatrix[n] = false;
		starMatrix[n] = newStarMatrix[n];
	}
	for (size_t n = 0; n < nOfRows; ++n)
	{
		coveredRows[n] = false;
	}
	/* move to step 2a */
	step2a(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases without forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal2(std::vector<int> &assignment, double &cost, const std::vector<double> &distMatrixIn, size_t nOfRows, size_t nOfColumns)
{
	/* make working copy of distance Matrix */
	m_distMatrix.assign(std::begin(distMatrixIn), std::end(distMatrixIn));

	/* recursively search for the minimum element and do the assignment */
	for (;;)
	{
		/* find minimum distance observation-to-track pair */
		double minValue = DBL_MAX;
		size_t tmpRow = 0;
		size_t tmpCol = 0;
		for (size_t row = 0; row < nOfRows; ++row)
		{
			for (size_t col = 0; col < nOfColumns; ++col)
			{
				const double value = m_distMatrix[row + nOfRows * col];
				if (value != DBL_MAX && (value < minValue))
				{
					minValue = value;
					tmpRow = row;
					tmpCol = col;
				}
			}
		}

		if (minValue != DBL_MAX)
		{
			assignment[tmpRow] = static_cast<int>(tmpCol);
			cost += minValue;
			for (size_t n = 0; n < nOfRows; ++n)
			{
				m_distMatrix[n + nOfRows * tmpCol] = DBL_MAX;
			}
			for (size_t n = 0; n < nOfColumns; ++n)
			{
				m_distMatrix[tmpRow + nOfRows * n] = DBL_MAX;
			}
		}
		else
		{
			break;
		}
	}
}
// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases with many forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal1(std::vector<int> &assignment, double &cost, const std::vector<double> &distMatrixIn, size_t nOfRows, size_t nOfColumns)
{
	/* make working copy of distance Matrix */
	m_distMatrix.assign(std::begin(distMatrixIn), std::end(distMatrixIn));

	/* allocate memory */
	int *nOfValidObservations = (int *)calloc(nOfRows, sizeof(int));
	int *nOfValidTracks = (int *)calloc(nOfColumns, sizeof(int));

	/* compute number of validations */
	bool infiniteValueFound = false;
	bool finiteValueFound = false;
	for (size_t row = 0; row < nOfRows; ++row)
	{
		for (size_t col = 0; col < nOfColumns; ++col)
		{
			if (m_distMatrix[row + nOfRows * col] != DBL_MAX)
			{
				nOfValidTracks[col] += 1;
				nOfValidObservations[row] += 1;
				finiteValueFound = true;
			}
			else
			{
				infiniteValueFound = true;
			}
		}
	}

	if (infiniteValueFound)
	{
		if (!finiteValueFound)
		{
			/* free allocated memory */
			free(nOfValidObservations);
			free(nOfValidTracks);
			return;
		}
		bool repeatSteps = true;

		while (repeatSteps)
		{
			repeatSteps = false;

			/* step 1: reject assignments of multiply validated tracks to singly validated observations		 */
			for (size_t col = 0; col < nOfColumns; ++col)
			{
				bool singleValidationFound = false;
				for (size_t row = 0; row < nOfRows; ++row)
				{
					if (m_distMatrix[row + nOfRows * col] != DBL_MAX && (nOfValidObservations[row] == 1))
					{
						singleValidationFound = true;
						break;
					}
				}
				if (singleValidationFound)
				{
					for (size_t nestedRow = 0; nestedRow < nOfRows; ++nestedRow)
						if ((nOfValidObservations[nestedRow] > 1) && m_distMatrix[nestedRow + nOfRows * col] != DBL_MAX)
						{
							m_distMatrix[nestedRow + nOfRows * col] = DBL_MAX;
							nOfValidObservations[nestedRow] -= 1;
							nOfValidTracks[col] -= 1;
							repeatSteps = true;
						}
				}
			}

			/* step 2: reject assignments of multiply validated observations to singly validated tracks */
			if (nOfColumns > 1)
			{
				for (size_t row = 0; row < nOfRows; ++row)
				{
					bool singleValidationFound = false;
					for (size_t col = 0; col < nOfColumns; ++col)
					{
						if (m_distMatrix[row + nOfRows * col] != DBL_MAX && (nOfValidTracks[col] == 1))
						{
							singleValidationFound = true;
							break;
						}
					}

					if (singleValidationFound)
					{
						for (size_t col = 0; col < nOfColumns; ++col)
						{
							if ((nOfValidTracks[col] > 1) && m_distMatrix[row + nOfRows * col] != DBL_MAX)
							{
								m_distMatrix[row + nOfRows * col] = DBL_MAX;
								nOfValidObservations[row] -= 1;
								nOfValidTracks[col] -= 1;
								repeatSteps = true;
							}
						}
					}
				}
			}
		} /* while(repeatSteps) */

		/* for each multiply validated track that validates only with singly validated  */
		/* observations, choose the observation with minimum distance */
		for (size_t row = 0; row < nOfRows; ++row)
		{
			if (nOfValidObservations[row] > 1)
			{
				bool allSinglyValidated = true;
				double minValue = DBL_MAX;
				size_t tmpCol = 0;
				for (size_t col = 0; col < nOfColumns; ++col)
				{
					const double value = m_distMatrix[row + nOfRows * col];
					if (value != DBL_MAX)
					{
						if (nOfValidTracks[col] > 1)
						{
							allSinglyValidated = false;
							break;
						}
						else if ((nOfValidTracks[col] == 1) && (value < minValue))
						{
							tmpCol = col;
							minValue = value;
						}
					}
				}

				if (allSinglyValidated)
				{
					assignment[row] = static_cast<int>(tmpCol);
					cost += minValue;
					for (size_t n = 0; n < nOfRows; ++n)
					{
						m_distMatrix[n + nOfRows * tmpCol] = DBL_MAX;
					}
					for (size_t n = 0; n < nOfColumns; ++n)
					{
						m_distMatrix[row + nOfRows * n] = DBL_MAX;
					}
				}
			}
		}

		// for each multiply validated observation that validates only with singly validated  track, choose the track with minimum distance
		for (size_t col = 0; col < nOfColumns; ++col)
		{
			if (nOfValidTracks[col] > 1)
			{
				bool allSinglyValidated = true;
				double minValue = DBL_MAX;
				size_t tmpRow = 0;
				for (size_t row = 0; row < nOfRows; ++row)
				{
					const double value = m_distMatrix[row + nOfRows * col];
					if (value != DBL_MAX)
					{
						if (nOfValidObservations[row] > 1)
						{
							allSinglyValidated = false;
							break;
						}
						else if ((nOfValidObservations[row] == 1) && (value < minValue))
						{
							tmpRow = row;
							minValue = value;
						}
					}
				}

				if (allSinglyValidated)
				{
					assignment[tmpRow] = static_cast<int>(col);
					cost += minValue;
					for (size_t n = 0; n < nOfRows; ++n)
					{
						m_distMatrix[n + nOfRows * col] = DBL_MAX;
					}
					for (size_t n = 0; n < nOfColumns; ++n)
					{
						m_distMatrix[tmpRow + nOfRows * n] = DBL_MAX;
					}
				}
			}
		}
	} /* if(infiniteValueFound) */

	/* now, recursively search for the minimum element and do the assignment */
	for (;;)
	{
		/* find minimum distance observation-to-track pair */
		double minValue = DBL_MAX;
		size_t tmpRow = 0;
		size_t tmpCol = 0;
		for (size_t row = 0; row < nOfRows; ++row)
		{
			for (size_t col = 0; col < nOfColumns; ++col)
			{
				const double value = m_distMatrix[row + nOfRows * col];
				if (value != DBL_MAX && (value < minValue))
				{
					minValue = value;
					tmpRow = row;
					tmpCol = col;
				}
			}
		}

		if (minValue != DBL_MAX)
		{
			assignment[tmpRow] = static_cast<int>(tmpCol);
			cost += minValue;
			for (size_t n = 0; n < nOfRows; ++n)
			{
				m_distMatrix[n + nOfRows * tmpCol] = DBL_MAX;
			}
			for (size_t n = 0; n < nOfColumns; ++n)
			{
				m_distMatrix[tmpRow + nOfRows * n] = DBL_MAX;
			}
		}
		else
		{
			break;
		}
	}

	/* free allocated memory */
	free(nOfValidObservations);
	free(nOfValidTracks);
}

void AssignmentProblemSolver::calcEmbeddins(std::vector<RegionEmbedding> &regionEmbeddings, const std::vector<sensor_msgs::obj_box_msg> &regions, cv::Mat currFrame)
{
	if (!regions.empty())
	{
		regionEmbeddings.resize(regions.size());
		// Bhatacharia distance between histograms
		if (1)
		{
			for (size_t j = 0; j < regions.size(); ++j)
			{
				int bins = 64;
				std::vector<int> histSize;
				std::vector<float> ranges;
				std::vector<int> channels;

				for (int i = 0, stop = currFrame.channels(); i < stop; ++i)
				{
					histSize.push_back(bins);
					ranges.push_back(0);
					ranges.push_back(255);
					channels.push_back(i);
				}

				cv::Rect rect_tmp = regions[j].bbox2d & cv::Rect(0, 0, currFrame.cols, currFrame.rows);
				std::vector<cv::Mat> regROI = {currFrame(rect_tmp)};
				cv::calcHist(regROI, channels, cv::Mat(), regionEmbeddings[j].m_hist, histSize, ranges, false);
				cv::normalize(regionEmbeddings[j].m_hist, regionEmbeddings[j].m_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
			}
		}
	}
}
// cv::RotatedRect AssignmentProblemSolver::calcPredictionEllipse(cv::Size_<double> minRadius) const
//{
//	// Move ellipse to velocity
//	//cv::Vec<double, 2> velocity;
//	//Point_t d(3.f * velocity[0], 3.f * velocity[1]);
//	cv::Vec<double, 2> velocity;
//	Point_t d(3.f * 0, 3.f * 0);
//
//	cv::RotatedRect rrect(m_predictionPoint, cv::Size2f(std::max(minRadius.width, fabs(d.x)), std::max(minRadius.height, fabs(d.y))), 0);
//
//	if (fabs(d.x) + fabs(d.y) > 4) // pix
//	{
//		if (fabs(d.x) > 0.0001f)
//		{
//			track_t l = std::min(rrect.size.width, rrect.size.height) / 3;
//
//			track_t p2_l = sqrtf(sqr(d.x) + sqr(d.y));
//			rrect.center.x = l * d.x / p2_l + m_predictionPoint.x;
//			rrect.center.y = l * d.y / p2_l + m_predictionPoint.y;
//
//			rrect.angle = atanf(d.y / d.x);
//		}
//		else
//		{
//			rrect.center.y += d.y / 3;
//			rrect.angle = static_cast<float>(CV_PI / 2.);
//		}
//	}
//	return rrect;
// }

double AssignmentProblemSolver::getPolygonArea(std::vector<cv::Point2f> const &points)
{
	const int sizep = points.size();
	if (sizep < 3)
		return 0.0;

	double area = points.back().x * points[0].y - points[0].x * points.back().y;
	for (int i = 1, v = 0; i < sizep; i++, v++)
	{
		area += (points[v].x * points[i].y);
		area -= (points[i].x * points[v].y);
	}

	return fabs(0.5 * area);
}
// void AssignmentProblemSolver::createDistaceMatrixByIoU(const std::vector<sensor_msgs::obj_box_msg> &regions_last, 
// 	const std::vector<sensor_msgs::obj_box_msg> &regions, const std::vector<double> &conf, std::vector<double> &costMatrix)
void AssignmentProblemSolver::createDistaceMatrixByIoU(const std::vector<sensor_msgs::obj_box_msg> &regions_last, 
	const std::vector<sensor_msgs::obj_box_msg> &regions, std::vector<double> &costMatrix)
{
	const size_t N = regions_last.size(); // Tracking objects
	for (int i = 0; i < regions_last.size(); ++i)
	{
		for (int j = 0; j < regions.size(); ++j)
		{
			// double intArea_2d = static_cast<double>((regions[j].bbox2d & regions_last[i].bbox2d).area());
			// double unionArea_2d = static_cast<double>(regions[j].bbox2d.area() + regions_last[i].bbox2d.area() - intArea_2d);
			// double dist = 1 - intArea_2d / unionArea_2d;
			// costMatrix[i + j * N] = dist;

			// if (dist > maxCost)
			//	maxCost = dist;

			std::vector<Eigen::Vector3d> trk_8_corner = regions_last[i].corners_in_local;
			std::vector<Eigen::Vector3d> det_8_corner = regions[j].corners_in_local;

			std::vector<cv::Point2f> trk_8_corner_in_plane;
			std::vector<cv::Point2f> det_8_corner_in_plane;
			for (int k = 0; k < 4; ++k)
			{
				trk_8_corner_in_plane.push_back(cv::Point2f(trk_8_corner[k][0], trk_8_corner[k][1]));
				det_8_corner_in_plane.push_back(cv::Point2f(det_8_corner[k][0], det_8_corner[k][1]));
			}

			cv::RotatedRect rect1 = cv::minAreaRect(trk_8_corner_in_plane);
			cv::RotatedRect rect2 = cv::minAreaRect(det_8_corner_in_plane);
			vector<cv::Point2f> intersectingRegion;
			cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion);
			// cv::RotatedRect rect3 = cv::minAreaRect(intersectingRegion);
			double intArea;
			if (intersectingRegion.size() < 3)
				intArea = 0;
			else
			{
				std::vector<cv::Point2f> order_pts;
				cv::convexHull(cv::Mat(intersectingRegion), order_pts, true);
				intArea = cv::contourArea(order_pts);

				// double area0 = contourArea(intersectingRegion);
				// std::vector<cv::Point2f> approx;
				// cv::approxPolyDP(intersectingRegion, approx, 5, true);
				// double Area = cv::contourArea(approx);
				// std::cout << "area: " << inter_area << std::endl;
				// // intArea = rect3.size.area();
				// intArea = cv::contourArea(intersectingRegion);
				// intArea = getPolygonArea(intersectingRegion);
			}

			double unionArea = static_cast<double>(cv::contourArea(trk_8_corner_in_plane) + cv::contourArea(det_8_corner_in_plane) - intArea);

			double iou_2d = 1 - intArea / unionArea;

			double z_max = min(trk_8_corner[4][2], det_8_corner[4][2]);
			double z_min = max(trk_8_corner[0][2], det_8_corner[0][2]);
			double inter_vol = intArea * max(0.0, z_max - z_min);
			double vol1 = regions_last[i].bbox3d_in_local.dimensions.x * regions_last[i].bbox3d_in_local.dimensions.y * regions_last[i].bbox3d_in_local.dimensions.z;
			double vol2 = regions[j].bbox3d_in_local.dimensions.x * regions[j].bbox3d_in_local.dimensions.y * regions[j].bbox3d_in_local.dimensions.z;

			// if (intArea!= 0)
			// {
			// 	std::cout << "intersectingRegion0: " << intersectingRegion[0].x << " " << intersectingRegion[0].y << std::endl;
			// 	std::cout << "intersectingRegion1: " << intersectingRegion[1].x << " " << intersectingRegion[1].y << std::endl;
			// 	std::cout << "intersectingRegion2: " << intersectingRegion[2].x << " " << intersectingRegion[2].y << std::endl;
			// 	std::cout << "intersectingRegion3: " << intersectingRegion[3].x << " " << intersectingRegion[3].y << std::endl;
			// 	std::cout << "intArea: " << intArea << " " << z_max - z_min << std::endl;
			// 	std::cout << "inter_vol: " << inter_vol << std::endl;
			// }
			double iou = 1 - inter_vol / (vol1 + vol2 - inter_vol);
			if (iou > 1 || iou < 0)
			{
				std::cout << "inter_vol: " << inter_vol << std::endl;
				std::cout << "vol1: " << vol1 << std::endl;
				std::cout << "vol2: " << vol2 << std::endl;
				std::cout << "iou: " << iou << std::endl;
			}
			if (iou > 1)
				iou = 1;
			if (iou < 0)
				iou = 1;
			if (isnan(iou))
				iou = 1;
			// costMatrix[i + j * N] = exp(-conf[i]) * iou;
			// costMatrix[i + j * N] = iou;
			costMatrix[i + j * N] = iou_2d;

			// if (iou > maxCost)
			//		maxCost = iou;
		}
	}
}

void AssignmentProblemSolver::createDistaceMatrixByDistance(const std::vector<sensor_msgs::obj_box_msg> &regions_last, const std::vector<sensor_msgs::obj_box_msg> &regions, std::vector<double> &costMatrix)
{
	const size_t N = regions_last.size(); // Tracking objects
	for (int i = 0; i < regions_last.size(); ++i)
	{
		for (int j = 0; j < regions.size(); ++j)
		{
			Eigen::Vector3d trk_center = Eigen::Vector3d(regions_last[i].bbox3d_in_local.pose.position.x,
														 regions_last[i].bbox3d_in_local.pose.position.y, regions_last[i].bbox3d_in_local.pose.position.z);
			Eigen::Vector3d det_center = Eigen::Vector3d(regions[j].bbox3d_in_local.pose.position.x,
														 regions[j].bbox3d_in_local.pose.position.y, regions[j].bbox3d_in_local.pose.position.z);
			double distance = (trk_center - det_center).norm();

			costMatrix[i + j * N] = distance;
		}
	}
}

void AssignmentProblemSolver::createDistaceMatrixByGIoU(const std::vector<sensor_msgs::obj_box_msg> &regions_last, 
    const std::vector<sensor_msgs::obj_box_msg> &regions, std::vector<double> &costMatrix)
{
    const size_t N = regions_last.size(); // Tracking objects
    for (int i = 0; i < regions_last.size(); ++i)
    {
        for (int j = 0; j < regions.size(); ++j)
        {
            std::vector<Eigen::Vector3d> trk_8_corner = regions_last[i].corners_in_local;
            std::vector<Eigen::Vector3d> det_8_corner = regions[j].corners_in_local;

            // 1. 计算3D IoU (基于你的现有代码)
            std::vector<cv::Point2f> trk_8_corner_in_plane;
            std::vector<cv::Point2f> det_8_corner_in_plane;
			std::vector<cv::Point2f> union_corner_in_plane;
            for (int k = 0; k < 4; ++k)
            {
                trk_8_corner_in_plane.push_back(cv::Point2f(trk_8_corner[k][0], trk_8_corner[k][1]));
                det_8_corner_in_plane.push_back(cv::Point2f(det_8_corner[k][0], det_8_corner[k][1]));
				union_corner_in_plane.push_back(cv::Point2f(trk_8_corner[k][0], trk_8_corner[k][1]));
				union_corner_in_plane.push_back(cv::Point2f(det_8_corner[k][0], det_8_corner[k][1]));
            }

            cv::RotatedRect rect1 = cv::minAreaRect(trk_8_corner_in_plane);
            cv::RotatedRect rect2 = cv::minAreaRect(det_8_corner_in_plane);
            std::vector<cv::Point2f> intersectingRegion;
            cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion);
            
            double intArea2D = 0;
            if (intersectingRegion.size() >= 3)
            {
                std::vector<cv::Point2f> order_pts;
                cv::convexHull(cv::Mat(intersectingRegion), order_pts, true);
                intArea2D = cv::contourArea(order_pts);
            }

			std::vector<cv::Point2f> union_order_pts;
			cv::convexHull(cv::Mat(union_corner_in_plane), union_order_pts, true);
			double unionArea2D = cv::contourArea(union_order_pts);

            double inter_z_max = std::min(trk_8_corner[4][2], det_8_corner[4][2]);
            double inter_z_min = std::max(trk_8_corner[0][2], det_8_corner[0][2]);

			double union_z_max = std::max(trk_8_corner[4][2], det_8_corner[4][2]);
			double union_z_min = std::min(trk_8_corner[0][2], det_8_corner[0][2]);

			double inter_height = std::max(0.0, inter_z_max - inter_z_min);
			double union_height = std::max(0.0, union_z_max - union_z_min);

            double inter_vol = intArea2D * inter_height;
			
			double min_enclosing_vol = unionArea2D * union_height;

			double vol1 = regions_last[i].bbox3d_in_local.dimensions.x * 
                         regions_last[i].bbox3d_in_local.dimensions.y * 
                         regions_last[i].bbox3d_in_local.dimensions.z;
            double vol2 = regions[j].bbox3d_in_local.dimensions.x * 
                         regions[j].bbox3d_in_local.dimensions.y * 
                         regions[j].bbox3d_in_local.dimensions.z;
            
            double union_vol = vol1 + vol2 - inter_vol;
            double iou_3d = (union_vol > 0) ? (inter_vol / union_vol) : 0.0;

            double giou_3d = -1.0;
            if (min_enclosing_vol > 0) {
                giou_3d = inter_vol / union_vol - (min_enclosing_vol - union_vol) / min_enclosing_vol;
            }

			// std::cout << "inter_height: " << inter_height << std::endl;
			// std::cout << "union_height: " << union_height << std::endl;
			// std::cout << "inter_vol: " << inter_vol << std::endl;
			// std::cout << "min_enclosing_vol: " << min_enclosing_vol << std::endl;

            // 处理异常值
            if (std::isnan(giou_3d) || std::isinf(giou_3d)) {
				// std::cout << "giou_3d: " << giou_3d << std::endl;
                giou_3d = -1.0; // 最大代价
            }

            costMatrix[i + j * N] = -giou_3d + 1;
        }
    }
}


double AssignmentProblemSolver::computeMinEnclosingVolume(const std::vector<Eigen::Vector3d>& corners1, 
                                                         const std::vector<Eigen::Vector3d>& corners2)
{
    // 找到所有角点的最小和最大坐标
    double min_x = corners1[0][0], max_x = corners1[0][0];
    double min_y = corners1[0][1], max_y = corners1[0][1];
    double min_z = corners1[0][2], max_z = corners1[0][2];

    // 检查第一个框的所有角点
    for (const auto& corner : corners1) {
        min_x = std::min(min_x, corner[0]);
        max_x = std::max(max_x, corner[0]);
        min_y = std::min(min_y, corner[1]);
        max_y = std::max(max_y, corner[1]);
        min_z = std::min(min_z, corner[2]);
        max_z = std::max(max_z, corner[2]);
    }

    // 检查第二个框的所有角点
    for (const auto& corner : corners2) {
        min_x = std::min(min_x, corner[0]);
        max_x = std::max(max_x, corner[0]);
        min_y = std::min(min_y, corner[1]);
        max_y = std::max(max_y, corner[1]);
        min_z = std::min(min_z, corner[2]);
        max_z = std::max(max_z, corner[2]);
    }

    // 计算最小闭合框体积
    double length = max_x - min_x;
                                                         
    double width = max_y - min_y;
    double height = max_z - min_z;

    return length * width * height;
}

