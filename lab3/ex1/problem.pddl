(define (domain parcel-delivery)
  (:requirements
    :strips
    :typing
    :negative-preconditions
    :durative-actions
    :action-costs
    :equality)

  (:types
    parcel location vehicle - object
    truck airplane boat - vehicle
    transport-type - object
  )

  (:constants
    road air water - transport-type
  )

  (:predicates
    (at ?x - (either parcel vehicle) ?l - location)
    (loaded ?p - parcel ?v - vehicle)
    (available ?v - vehicle)
    (delivered ?p - parcel)
    (connected ?l1 ?l2 - location ?mode - transport-type)
  )

  (:functions
    (road-cost ?l1 ?l2 - location)
    (air-cost ?l1 ?l2 - location)
    (water-cost ?l1 ?l2 - location)
    (road-time ?l1 ?l2 - location)
    (air-time ?l1 ?l2 - location)
    (water-time ?l1 ?l2 - location)
    (total-cost)
  )

  ;; Loading action
  (:durative-action load
    :parameters (?p - parcel ?v - vehicle ?l - location)
    :duration (= ?duration 1)
    :condition (and
      (at start (at ?p ?l))
      (at start (at ?v ?l))
      (at start (available ?v)))
    :effect (and
      (at start (not (at ?p ?l)))
      (at start (not (available ?v)))
      (at end (loaded ?p ?v))
      (at start (increase (total-cost) 10)))
  )

  ;; Unloading action
  (:durative-action unload
    :parameters (?p - parcel ?v - vehicle ?l - location)
    :duration (= ?duration 1)
    :condition (and
      (at start (loaded ?p ?v))
      (at start (at ?v ?l)))
    :effect (and
      (at start (not (loaded ?p ?v)))
      (at end (at ?p ?l))
      (at end (available ?v))
      (at end (delivered ?p))
      (at start (increase (total-cost) 10)))
  )

  ;; Road transport
  (:durative-action drive
    :parameters (?v - truck ?from ?to - location)
    :duration (= ?duration (road-time ?from ?to))
    :condition (and
      (at start (at ?v ?from))
      (over all (connected ?from ?to road)))
    :effect (and
      (at start (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at start (increase (total-cost) (road-cost ?from ?to))))
  )

  ;; Air transport
  (:durative-action fly
    :parameters (?v - airplane ?from ?to - location)
    :duration (= ?duration (air-time ?from ?to))
    :condition (and
      (at start (at ?v ?from))
      (over all (connected ?from ?to air)))
    :effect (and
      (at start (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at start (increase (total-cost) (air-cost ?from ?to))))
  )

  ;; Water transport
  (:durative-action sail
    :parameters (?v - boat ?from ?to - location)
    :duration (= ?duration (water-time ?from ?to))
    :condition (and
      (at start (at ?v ?from))
      (over all (connected ?from ?to water)))
    :effect (and
      (at start (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at start (increase (total-cost) (water-cost ?from ?to))))
  )
)