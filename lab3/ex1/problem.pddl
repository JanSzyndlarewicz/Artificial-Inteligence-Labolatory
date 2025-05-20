(define (problem deliver-packages)
  (:domain package-transport)
  (:objects
    locA locB locC - location
    truck1 - vehicle
    pkg1 pkg2 - package
  )
  (:init
    (at pkg1 locA) (at pkg2 locA)
    (vehicle-at truck1 locA)  ; Updated here
    (connected locA locB) (connected locB locC)
    (= (distance locA locB) 10) (= (distance locB locC) 15)
    (= (fuel-cost) 0)
  )
  (:goal (and (at pkg1 locC) (at pkg2 locB)))
  (:metric minimize (fuel-cost))
)