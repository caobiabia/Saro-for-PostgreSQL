select  count(*) from votes as v,  		posts as p,          users as u where v.UserId = p.OwnerUserId 	and p.OwnerUserId = u.Id  AND p.Score<=44  AND p.ViewCount<=5271  AND u.Reputation<=267;
