//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MultiSTKFieldContainer.hpp"
#include "Albany_MultiSTKFieldContainer_Def.hpp"

namespace Albany {

template class MultiSTKFieldContainer<true>;
template class MultiSTKFieldContainer<false>;

}  // namespace Albany
