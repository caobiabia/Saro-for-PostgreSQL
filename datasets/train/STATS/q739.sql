select  count(*) from votes as v,  		posts as p,          users as u where v.UserId = p.OwnerUserId 	and p.OwnerUserId = u.Id  AND u.UpVotes<=18;
