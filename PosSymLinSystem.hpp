#ifndef POS_SYM_LIN_SYSTEM_HPP
#define POS_SYM_LIN_SYSTEM_HPP

#include "LinearSystem.hpp"

class PosSymLinSystem : public LinearSystem {
public:
    // Inherit base‐class constructor
    using LinearSystem::LinearSystem;

    virtual ~PosSymLinSystem() = default;

    // Override Solve() to use Conjugate Gradient
    Vector Solve() const override;
};

#endif // POS_SYM_LIN_SYSTEM_HPP
