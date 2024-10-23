select  count(*) from postHistory as ph,          votes as v,  		users as u,  		badges as b  where u.Id = ph.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND v.VoteTypeId=2;
